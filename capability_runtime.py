"""Typed capability policy, registry, and bounded execution helpers for local task control."""

from __future__ import annotations

import asyncio
import ctypes
import os
import re
import shutil
import subprocess
import tempfile
import time
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import urlparse

from config import APP_CONFIG, AppConfig
from data_structures import (
    CapabilityAuditRecord,
    CapabilityAvailabilityStatus,
    CapabilityExecutionResult,
    CapabilityExecutionStatus,
    CapabilityPolicyDecision,
    CapabilityPolicyOutcome,
    CapabilityRegistration,
    CapabilityRegistryView,
    CapabilityRequest,
    CapabilityType,
    UserSettingsProfile,
    utc_now,
)

if TYPE_CHECKING:
    from model_manager import ModelHealthSnapshot


_CAPABILITY_SUMMARIES: dict[CapabilityType, str] = {
    CapabilityType.FILE_OPERATION: "Bounded file operations inside allowlisted roots.",
    CapabilityType.SHELL_COMMAND: "Allowlisted shell commands for local maintenance workflows.",
    CapabilityType.BROWSER_ACTION: "Allowlisted browser actions constrained to approved domains.",
    CapabilityType.APP_WINDOW_FOCUS: "Visible app or window focus changes for allowlisted targets.",
    CapabilityType.CLIPBOARD_ACTION: "Clipboard read or write actions with explicit policy checks.",
    CapabilityType.SCREENSHOT: "On-demand screenshot capture with bounded output paths.",
    CapabilityType.OCR_REQUEST: "CPU-first OCR requests against explicitly provided image inputs.",
    CapabilityType.DESKTOP_INPUT: "Approval-gated desktop mouse or keyboard input actions.",
}

_SUPPORTED_ACTIONS: dict[CapabilityType, tuple[str, ...]] = {
    CapabilityType.FILE_OPERATION: ("read", "write", "move", "copy", "archive", "delete"),
    CapabilityType.SHELL_COMMAND: ("run",),
    CapabilityType.BROWSER_ACTION: ("navigate", "read", "click", "type", "download"),
    CapabilityType.APP_WINDOW_FOCUS: ("focus_window",),
    CapabilityType.CLIPBOARD_ACTION: ("read", "write", "clear"),
    CapabilityType.SCREENSHOT: ("capture_screenshot",),
    CapabilityType.OCR_REQUEST: ("ocr_image",),
    CapabilityType.DESKTOP_INPUT: ("type_text", "press_keys", "mouse_click", "mouse_move"),
}

_MAX_TEXT_FILE_BYTES = 64 * 1024
_MAX_DIRECTORY_LISTING_ENTRIES = 128
_MAX_SHELL_OUTPUT_BYTES = 32 * 1024
_MAX_OCR_TEXT_CHARS = 4096
_WINDOW_MATCH_TIMEOUT_S = 5.0
_WINDOW_MATCH_POLL_S = 0.1
_MAX_MOUSE_MOVE_STEPS = 24
_MAX_DESKTOP_CLICK_COUNT = 3


@dataclass(slots=True, frozen=True)
class WindowSnapshot:
    """Visible top-level window snapshot used for bounded app/browser validation."""

    hwnd: int
    title: str
    process_name: str
    pid: int


@dataclass(slots=True)
class CapabilityPolicyEngine:
    """Evaluate typed capability requests against local-first policy and settings."""

    config: AppConfig = APP_CONFIG
    workspace_root: Path = Path.cwd()

    def build_registry_view(
        self,
        *,
        profile: UserSettingsProfile,
        snapshot: ModelHealthSnapshot,
        recent_decisions: tuple[CapabilityPolicyDecision, ...] = (),
        recent_audits: tuple[CapabilityAuditRecord, ...] = (),
    ) -> CapabilityRegistryView:
        registrations = tuple(
            self._build_registration(capability_type=capability_type, profile=profile, snapshot=snapshot)
            for capability_type in CapabilityType
        )
        return CapabilityRegistryView(
            registrations=registrations,
            recent_decisions=recent_decisions,
            recent_audits=recent_audits,
            updated_at=utc_now(),
        )

    def evaluate(
        self,
        request: CapabilityRequest,
        *,
        profile: UserSettingsProfile,
        snapshot: ModelHealthSnapshot,
    ) -> CapabilityPolicyDecision:
        reason_codes: list[str] = []
        warnings: list[str] = []
        desktop_settings = dict(profile.desktop)
        desktop_enabled = bool(desktop_settings.get("enabled", False))
        approval_policy = str(desktop_settings.get("approval_policy", "approve_risky_only"))
        enabled_capabilities = {
            str(item)
            for item in desktop_settings.get("enabled_capabilities", ())
            if str(item).strip()
        }
        observation_tier = str(profile.observation.get("tier", "screenshot_on_demand"))
        resource_pressure = self._resource_pressure(snapshot)

        # Block dangerous flags regardless of policy
        if request.requires_elevation:
            reason_codes.append("admin_elevation_blocked")
        if request.persistent_background:
            reason_codes.append("startup_persistence_blocked")
        if request.hidden_execution:
            reason_codes.append("hidden_background_control_blocked")
        if request.touches_credentials:
            reason_codes.append("credential_harvesting_blocked")
        if request.unrestricted_scope:
            reason_codes.append("unrestricted_scope_blocked")
        if reason_codes:
            return self._decision(
                request=request,
                outcome=CapabilityPolicyOutcome.DENIED,
                availability=CapabilityAvailabilityStatus.DENIED_BY_POLICY,
                reason_codes=tuple(reason_codes),
                detail="Dangerous capability flags are blocked by default.",
            )
        if not desktop_enabled:
            return self._decision(
                request=request,
                outcome=CapabilityPolicyOutcome.DENIED,
                availability=CapabilityAvailabilityStatus.AVAILABLE,
                reason_codes=("desktop_mode_disabled",),
                detail="Desktop task capabilities stay disabled until the operator enables desktop mode.",
            )
        if request.capability_type.value not in enabled_capabilities:
            return self._decision(
                request=request,
                outcome=CapabilityPolicyOutcome.DENIED,
                availability=CapabilityAvailabilityStatus.AVAILABLE,
                reason_codes=("capability_not_enabled",),
                detail="This capability is supported but not enabled in the active settings profile.",
            )

        # Desktop approval policy enforcement
        if approval_policy == "manual_only":
            # All desktop/file/shell actions require approval
            if request.capability_type in {
                CapabilityType.FILE_OPERATION,
                CapabilityType.SHELL_COMMAND,
                CapabilityType.BROWSER_ACTION,
                CapabilityType.APP_WINDOW_FOCUS,
                CapabilityType.CLIPBOARD_ACTION,
                CapabilityType.SCREENSHOT,
                CapabilityType.OCR_REQUEST,
                CapabilityType.DESKTOP_INPUT,
            }:
                return self._decision(
                    request=request,
                    outcome=CapabilityPolicyOutcome.REQUIRES_APPROVAL,
                    availability=CapabilityAvailabilityStatus.REQUIRES_APPROVAL,
                    reason_codes=("manual_policy_all_require_approval",),
                    detail="Manual-only policy: all desktop actions require explicit approval.",
                )
        elif approval_policy == "safe_auto":
            # All desktop/file/shell actions are auto-approved
            if request.capability_type in {
                CapabilityType.FILE_OPERATION,
                CapabilityType.SHELL_COMMAND,
                CapabilityType.BROWSER_ACTION,
                CapabilityType.APP_WINDOW_FOCUS,
                CapabilityType.CLIPBOARD_ACTION,
                CapabilityType.SCREENSHOT,
                CapabilityType.OCR_REQUEST,
                CapabilityType.DESKTOP_INPUT,
            }:
                return self._decision(
                    request=request,
                    outcome=CapabilityPolicyOutcome.ALLOWED,
                    availability=CapabilityAvailabilityStatus.AVAILABLE,
                    reason_codes=("safe_auto_policy_all_allowed",),
                    detail="Safe-auto policy: all desktop actions are auto-approved.",
                )
        # else: default (approve_risky_only) - fall through to existing logic
        if resource_pressure and request.capability_type in {
            CapabilityType.SCREENSHOT,
            CapabilityType.OCR_REQUEST,
            CapabilityType.DESKTOP_INPUT,
        }:
            warnings.append("resource_pressure")
            return self._decision(
                request=request,
                outcome=CapabilityPolicyOutcome.DEGRADED,
                availability=CapabilityAvailabilityStatus.DEGRADED,
                reason_codes=("resource_pressure",),
                detail="The runtime is under resource pressure, so the requested capability is degraded.",
                warnings=tuple(warnings),
            )

        if request.capability_type == CapabilityType.FILE_OPERATION and request.file_operation is not None:
            allowlist_reason = self._check_paths_in_allowlist(
                request.file_operation.source_path,
                request.file_operation.destination_path,
                allowlisted_roots=tuple(
                    str(item) for item in desktop_settings.get("allowlisted_roots", ())
                ),
            )
            if allowlist_reason is not None:
                return self._decision(
                    request=request,
                    outcome=CapabilityPolicyOutcome.DENIED,
                    availability=CapabilityAvailabilityStatus.DENIED_BY_POLICY,
                    reason_codes=(allowlist_reason,),
                    detail="File operations must stay inside explicit allowlisted roots.",
                )
            if request.destructive or request.file_operation.operation in {"archive", "delete"}:
                return self._decision(
                    request=request,
                    outcome=CapabilityPolicyOutcome.REQUIRES_APPROVAL,
                    availability=CapabilityAvailabilityStatus.REQUIRES_APPROVAL,
                    reason_codes=("destructive_file_operation",),
                    detail="Destructive or archival file changes require approval before execution.",
                )
            return self._decision(
                request=request,
                outcome=CapabilityPolicyOutcome.ALLOWED,
                availability=CapabilityAvailabilityStatus.AVAILABLE,
                reason_codes=("allowlisted_file_operation",),
                detail="File operation stays inside approved workspace roots.",
            )

        if request.capability_type == CapabilityType.SHELL_COMMAND and request.shell_command is not None:
            command_name = request.shell_command.command.strip().split()[0]
            allowed_commands = {
                str(item)
                for item in desktop_settings.get("allowlisted_shell_commands", ())
                if str(item).strip()
            }
            if command_name not in allowed_commands:
                return self._decision(
                    request=request,
                    outcome=CapabilityPolicyOutcome.DENIED,
                    availability=CapabilityAvailabilityStatus.DENIED_BY_POLICY,
                    reason_codes=("command_not_allowlisted",),
                    detail="Shell commands must stay on the explicit allowlist.",
                )
            if request.shell_command.working_directory:
                allowlist_reason = self._check_paths_in_allowlist(
                    request.shell_command.working_directory,
                    allowlisted_roots=tuple(
                        str(item) for item in desktop_settings.get("allowlisted_roots", ())
                    ),
                )
                if allowlist_reason is not None:
                    return self._decision(
                        request=request,
                        outcome=CapabilityPolicyOutcome.DENIED,
                        availability=CapabilityAvailabilityStatus.DENIED_BY_POLICY,
                        reason_codes=(allowlist_reason,),
                        detail="Shell command working directories must stay inside approved roots.",
                    )
            if request.destructive or request.cross_app:
                return self._decision(
                    request=request,
                    outcome=CapabilityPolicyOutcome.REQUIRES_APPROVAL,
                    availability=CapabilityAvailabilityStatus.REQUIRES_APPROVAL,
                    reason_codes=("risky_shell_command",),
                    detail="Risky shell commands require approval before execution.",
                )
            return self._decision(
                request=request,
                outcome=CapabilityPolicyOutcome.ALLOWED,
                availability=CapabilityAvailabilityStatus.AVAILABLE,
                reason_codes=("allowlisted_shell_command",),
                detail="Shell command is allowlisted and scoped to the local workspace.",
            )

        if request.capability_type == CapabilityType.BROWSER_ACTION and request.browser_action is not None:
            domain = request.browser_action.domain or urlparse(request.browser_action.url).netloc.split(":")[0]
            if not domain:
                return self._decision(
                    request=request,
                    outcome=CapabilityPolicyOutcome.DENIED,
                    availability=CapabilityAvailabilityStatus.DENIED_BY_POLICY,
                    reason_codes=("missing_browser_domain",),
                    detail="Browser actions must resolve to an explicit allowlisted domain.",
                )
            allowed_domains = {
                str(item)
                for item in desktop_settings.get("allowlisted_browser_domains", ())
                if str(item).strip()
            }
            if domain not in allowed_domains:
                return self._decision(
                    request=request,
                    outcome=CapabilityPolicyOutcome.DENIED,
                    availability=CapabilityAvailabilityStatus.DENIED_BY_POLICY,
                    reason_codes=("browser_domain_not_allowlisted",),
                    detail="Browser actions must stay inside the approved domain allowlist.",
                )
            if request.browser_action.action in {"click", "type", "download"}:
                return self._decision(
                    request=request,
                    outcome=CapabilityPolicyOutcome.REQUIRES_APPROVAL,
                    availability=CapabilityAvailabilityStatus.REQUIRES_APPROVAL,
                    reason_codes=("interactive_browser_action",),
                    detail="Interactive browser actions require approval before execution.",
                )
            return self._decision(
                request=request,
                outcome=CapabilityPolicyOutcome.ALLOWED,
                availability=CapabilityAvailabilityStatus.AVAILABLE,
                reason_codes=("allowlisted_browser_action",),
                detail="Browser action is scoped to an allowlisted domain.",
            )

        if request.capability_type == CapabilityType.APP_WINDOW_FOCUS and request.app_focus is not None:
            allowed_apps = {
                str(item)
                for item in desktop_settings.get("allowlisted_apps", ())
                if str(item).strip()
            }
            if request.app_focus.app_name not in allowed_apps:
                return self._decision(
                    request=request,
                    outcome=CapabilityPolicyOutcome.DENIED,
                    availability=CapabilityAvailabilityStatus.DENIED_BY_POLICY,
                    reason_codes=("app_not_allowlisted",),
                    detail="App or window focus changes must stay inside the explicit app allowlist.",
                )
            return self._decision(
                request=request,
                outcome=CapabilityPolicyOutcome.ALLOWED,
                availability=CapabilityAvailabilityStatus.AVAILABLE,
                reason_codes=("allowlisted_app_focus",),
                detail="App focus change stays inside the explicit app allowlist.",
            )

        if request.capability_type == CapabilityType.CLIPBOARD_ACTION and request.clipboard_action is not None:
            if request.clipboard_action.action == "read":
                return self._decision(
                    request=request,
                    outcome=CapabilityPolicyOutcome.REQUIRES_APPROVAL,
                    availability=CapabilityAvailabilityStatus.REQUIRES_APPROVAL,
                    reason_codes=("clipboard_read_requires_approval",),
                    detail="Clipboard reads require approval before execution.",
                )
            return self._decision(
                request=request,
                outcome=CapabilityPolicyOutcome.ALLOWED,
                availability=CapabilityAvailabilityStatus.AVAILABLE,
                reason_codes=("clipboard_write_allowed",),
                detail="Clipboard write request passed bounded policy checks.",
            )

        if request.capability_type == CapabilityType.SCREENSHOT and request.screenshot is not None:
            allowlist_reason = self._check_paths_in_allowlist(
                request.screenshot.save_path,
                allowlisted_roots=tuple(
                    str(item) for item in desktop_settings.get("allowlisted_roots", ())
                ),
            )
            if allowlist_reason is not None:
                return self._decision(
                    request=request,
                    outcome=CapabilityPolicyOutcome.DENIED,
                    availability=CapabilityAvailabilityStatus.DENIED_BY_POLICY,
                    reason_codes=(allowlist_reason,),
                    detail="Screenshot outputs must stay inside approved roots.",
                )
            return self._decision(
                request=request,
                outcome=CapabilityPolicyOutcome.ALLOWED,
                availability=CapabilityAvailabilityStatus.AVAILABLE,
                reason_codes=("bounded_screenshot_capture",),
                detail="Screenshot request stays bounded to a local output path.",
            )

        if request.capability_type == CapabilityType.OCR_REQUEST and request.ocr_request is not None:
            if observation_tier not in {"ocr_on_step", "vision_on_step", "continuous_capture"}:
                return self._decision(
                    request=request,
                    outcome=CapabilityPolicyOutcome.DENIED,
                    availability=CapabilityAvailabilityStatus.DENIED_BY_POLICY,
                    reason_codes=("observation_tier_not_enabled",),
                    detail="OCR requests stay disabled until an OCR-capable observation tier is enabled.",
                )
            allowlist_reason = self._check_paths_in_allowlist(
                request.ocr_request.source_image_path,
                allowlisted_roots=tuple(
                    str(item) for item in desktop_settings.get("allowlisted_roots", ())
                ),
            )
            if allowlist_reason is not None:
                return self._decision(
                    request=request,
                    outcome=CapabilityPolicyOutcome.DENIED,
                    availability=CapabilityAvailabilityStatus.DENIED_BY_POLICY,
                    reason_codes=(allowlist_reason,),
                    detail="OCR inputs must stay inside approved roots.",
                )
            return self._decision(
                request=request,
                outcome=CapabilityPolicyOutcome.ALLOWED,
                availability=CapabilityAvailabilityStatus.AVAILABLE,
                reason_codes=("bounded_ocr_request",),
                detail="OCR request passed bounded path and observation-tier checks.",
            )

        if request.capability_type == CapabilityType.DESKTOP_INPUT and request.desktop_input is not None:
            allowed_apps = {
                str(item)
                for item in desktop_settings.get("allowlisted_apps", ())
                if str(item).strip()
            }
            if not request.desktop_input.target.strip():
                return self._decision(
                    request=request,
                    outcome=CapabilityPolicyOutcome.DENIED,
                    availability=CapabilityAvailabilityStatus.DENIED_BY_POLICY,
                    reason_codes=("desktop_target_required",),
                    detail="Desktop input actions require an explicit allowlisted target app or window.",
                )
            if request.desktop_input.target and request.desktop_input.target not in allowed_apps:
                return self._decision(
                    request=request,
                    outcome=CapabilityPolicyOutcome.DENIED,
                    availability=CapabilityAvailabilityStatus.DENIED_BY_POLICY,
                    reason_codes=("desktop_target_not_allowlisted",),
                    detail="Desktop input targets must stay inside the explicit app allowlist.",
                )
            return self._decision(
                request=request,
                outcome=CapabilityPolicyOutcome.REQUIRES_APPROVAL,
                availability=CapabilityAvailabilityStatus.REQUIRES_APPROVAL,
                reason_codes=("manual_input_requires_approval",),
                detail="Desktop input actions always require approval before execution.",
            )

        return self._decision(
            request=request,
            outcome=CapabilityPolicyOutcome.DENIED,
            availability=CapabilityAvailabilityStatus.UNAVAILABLE,
            reason_codes=("unsupported_capability_request",),
            detail="The requested capability payload is not supported by the current runtime.",
        )

    def _build_registration(
        self,
        *,
        capability_type: CapabilityType,
        profile: UserSettingsProfile,
        snapshot: ModelHealthSnapshot,
    ) -> CapabilityRegistration:
        desktop_settings = dict(profile.desktop)
        desktop_enabled = bool(desktop_settings.get("enabled", False))
        enabled_capabilities = {
            str(item)
            for item in desktop_settings.get("enabled_capabilities", ())
            if str(item).strip()
        }
        observation_tier = str(profile.observation.get("tier", "screenshot_on_demand"))
        enabled = desktop_enabled and capability_type.value in enabled_capabilities
        reason = "capability_not_enabled"
        detail = "Capability is supported but disabled in the active settings profile."
        status = CapabilityAvailabilityStatus.AVAILABLE
        default_policy_outcome = CapabilityPolicyOutcome.DENIED
        if not desktop_enabled:
            reason = "desktop_mode_disabled"
            detail = "Desktop task mode is disabled, so capability execution requests are denied."
        elif self._resource_pressure(snapshot) and capability_type in {
            CapabilityType.SCREENSHOT,
            CapabilityType.OCR_REQUEST,
            CapabilityType.DESKTOP_INPUT,
        }:
            status = CapabilityAvailabilityStatus.DEGRADED
            reason = "resource_pressure"
            detail = "Current runtime health indicates low headroom for visual or direct-input capabilities."
            default_policy_outcome = CapabilityPolicyOutcome.DEGRADED
        elif capability_type == CapabilityType.OCR_REQUEST and observation_tier not in {
            "ocr_on_step",
            "vision_on_step",
            "continuous_capture",
        }:
            status = CapabilityAvailabilityStatus.DENIED_BY_POLICY
            reason = "observation_tier_not_enabled"
            detail = "OCR remains disabled until an OCR-capable observation tier is explicitly enabled."
        elif enabled:
            if capability_type in {CapabilityType.CLIPBOARD_ACTION, CapabilityType.DESKTOP_INPUT}:
                status = CapabilityAvailabilityStatus.REQUIRES_APPROVAL
                reason = "approval_gated_capability"
                detail = "This capability is enabled but remains approval-gated by policy."
                default_policy_outcome = CapabilityPolicyOutcome.REQUIRES_APPROVAL
            else:
                status = CapabilityAvailabilityStatus.AVAILABLE
                reason = "enabled"
                detail = (
                    "Capability is enabled and ready for bounded live execution."
                    if self._executor_kind(capability_type) == "live"
                    else "Capability is enabled and ready for bounded stub execution."
                )
                default_policy_outcome = CapabilityPolicyOutcome.ALLOWED
        allowlisted_targets = self._allowlisted_targets(capability_type=capability_type, profile=profile)
        return CapabilityRegistration(
            capability_type=capability_type,
            summary=_CAPABILITY_SUMMARIES[capability_type],
            available=True,
            enabled=enabled,
            status=status,
            default_policy_outcome=default_policy_outcome,
            reason=reason,
            detail=detail,
            supported_actions=_SUPPORTED_ACTIONS[capability_type],
            allowlisted_targets=allowlisted_targets,
            executor_kind=self._executor_kind(capability_type),
            metadata={
                "desktop_enabled": desktop_enabled,
                "observation_tier": observation_tier,
            },
        )

    @staticmethod
    def _decision(
        *,
        request: CapabilityRequest,
        outcome: CapabilityPolicyOutcome,
        availability: CapabilityAvailabilityStatus,
        reason_codes: tuple[str, ...],
        detail: str,
        warnings: tuple[str, ...] = (),
    ) -> CapabilityPolicyDecision:
        return CapabilityPolicyDecision(
            request_id=request.request_id,
            capability_type=request.capability_type,
            action_name=request.action_name(),
            outcome=outcome,
            availability=availability,
            requires_approval=outcome == CapabilityPolicyOutcome.REQUIRES_APPROVAL,
            reason_codes=reason_codes,
            detail=detail,
            warnings=warnings,
            decided_at=utc_now(),
        )

    def _check_paths_in_allowlist(
        self,
        *paths: str,
        allowlisted_roots: tuple[str, ...],
    ) -> str | None:
        resolved_roots = tuple(self._resolve_root(root) for root in allowlisted_roots if str(root).strip())
        for raw_path in paths:
            candidate = str(raw_path).strip()
            if not candidate:
                continue
            resolved_candidate = self._resolve_root(candidate)
            if not any(resolved_candidate.is_relative_to(root) for root in resolved_roots):
                return "path_not_allowlisted"
        return None

    def _allowlisted_targets(
        self,
        *,
        capability_type: CapabilityType,
        profile: UserSettingsProfile,
    ) -> tuple[str, ...]:
        desktop_settings = dict(profile.desktop)
        if capability_type == CapabilityType.SHELL_COMMAND:
            return tuple(str(item) for item in desktop_settings.get("allowlisted_shell_commands", ()))
        if capability_type == CapabilityType.BROWSER_ACTION:
            return tuple(str(item) for item in desktop_settings.get("allowlisted_browser_domains", ()))
        if capability_type in {CapabilityType.APP_WINDOW_FOCUS, CapabilityType.DESKTOP_INPUT}:
            return tuple(str(item) for item in desktop_settings.get("allowlisted_apps", ()))
        if capability_type in {
            CapabilityType.FILE_OPERATION,
            CapabilityType.SCREENSHOT,
            CapabilityType.OCR_REQUEST,
        }:
            return tuple(str(item) for item in desktop_settings.get("allowlisted_roots", ()))
        return ()

    def _resolve_root(self, raw_path: str) -> Path:
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = self.workspace_root / candidate
        return candidate.resolve(strict=False)

    def _resource_pressure(self, snapshot: ModelHealthSnapshot) -> bool:
        if getattr(snapshot, "governor_active", False) and not getattr(snapshot, "allow_ocr_on_step", True):
            return True
        if snapshot.available_ram_gb is not None and snapshot.available_ram_gb <= self.config.backend_runtime.low_ram_headroom_gb:
            return True
        if snapshot.fallback_active:
            return True
        return False

    @staticmethod
    def _executor_kind(capability_type: CapabilityType) -> str:
        if capability_type in {
            CapabilityType.FILE_OPERATION,
            CapabilityType.SHELL_COMMAND,
            CapabilityType.BROWSER_ACTION,
            CapabilityType.APP_WINDOW_FOCUS,
            CapabilityType.SCREENSHOT,
            CapabilityType.OCR_REQUEST,
            CapabilityType.DESKTOP_INPUT,
        }:
            return "live"
        return "stub"


@dataclass(slots=True)
class CapabilityStubExecutor:
    """Return bounded stub execution results for typed capability requests."""

    def execute(
        self,
        request: CapabilityRequest,
        *,
        decision: CapabilityPolicyDecision,
    ) -> CapabilityExecutionResult:
        if decision.outcome == CapabilityPolicyOutcome.ALLOWED:
            return CapabilityExecutionResult(
                request_id=request.request_id,
                capability_type=request.capability_type,
                action_name=request.action_name(),
                status=CapabilityExecutionStatus.SUCCEEDED,
                summary=f"Stub executor accepted {request.capability_type.value} request '{request.action_name()}'.",
                detail="No live OS adapter ran; the stub executor only validated the typed request and policy path.",
                executor_kind="stub",
                output_ref=request.target_summary(),
                warnings=("stub_executor_only",),
                metadata={"target": request.target_summary(), "summary": request.summary},
                completed_at=utc_now(),
            )
        return CapabilityExecutionResult(
            request_id=request.request_id,
            capability_type=request.capability_type,
            action_name=request.action_name(),
            status=CapabilityExecutionStatus.BLOCKED,
            summary=f"Capability request '{request.action_name()}' did not run.",
            detail=decision.detail,
            executor_kind="stub",
            output_ref=request.target_summary(),
            warnings=decision.warnings,
            metadata={"reason_codes": list(decision.reason_codes)},
            completed_at=utc_now(),
        )


@dataclass(slots=True)
class CapabilityExecutor:
    """Run bounded live executors for typed local task and observation requests."""

    config: AppConfig = APP_CONFIG
    workspace_root: Path = Path.cwd()
    stub_executor: CapabilityStubExecutor = field(default_factory=CapabilityStubExecutor)
    max_text_file_bytes: int = _MAX_TEXT_FILE_BYTES
    max_directory_listing_entries: int = _MAX_DIRECTORY_LISTING_ENTRIES
    max_shell_output_bytes: int = _MAX_SHELL_OUTPUT_BYTES

    async def execute(
        self,
        request: CapabilityRequest,
        *,
        decision: CapabilityPolicyDecision,
        profile: UserSettingsProfile,
        should_abort: Callable[[], str | None] | None = None,
    ) -> CapabilityExecutionResult:
        if decision.outcome != CapabilityPolicyOutcome.ALLOWED:
            return self.stub_executor.execute(request, decision=decision)
        if request.capability_type == CapabilityType.FILE_OPERATION:
            try:
                return await asyncio.to_thread(
                    self._execute_file_operation,
                    request,
                    profile,
                )
            except Exception as exc:
                return self._failed_result(
                    request,
                    executor_kind="live_file",
                    detail=str(exc),
                    warnings=("live_file_execution_failed",),
                )
        if request.capability_type == CapabilityType.SHELL_COMMAND:
            try:
                return await asyncio.to_thread(
                    self._execute_shell_command,
                    request,
                    profile,
                )
            except Exception as exc:
                return self._failed_result(
                    request,
                    executor_kind="live_shell",
                    detail=str(exc),
                    warnings=("live_shell_execution_failed",),
                )
        if request.capability_type == CapabilityType.BROWSER_ACTION:
            try:
                return await asyncio.to_thread(
                    self._execute_browser_action,
                    request,
                )
            except Exception as exc:
                return self._failed_result(
                    request,
                    executor_kind="live_browser",
                    detail=str(exc),
                    warnings=("live_browser_execution_failed",),
                )
        if request.capability_type == CapabilityType.APP_WINDOW_FOCUS:
            try:
                return await asyncio.to_thread(
                    self._execute_app_focus,
                    request,
                )
            except Exception as exc:
                return self._failed_result(
                    request,
                    executor_kind="live_window",
                    detail=str(exc),
                    warnings=("live_window_focus_failed",),
                )
        if request.capability_type == CapabilityType.SCREENSHOT:
            try:
                return await asyncio.to_thread(
                    self._execute_screenshot,
                    request,
                    profile,
                )
            except Exception as exc:
                return self._failed_result(
                    request,
                    executor_kind="live_screenshot",
                    detail=str(exc),
                    warnings=("live_screenshot_failed",),
                )
        if request.capability_type == CapabilityType.OCR_REQUEST:
            try:
                return await asyncio.to_thread(
                    self._execute_ocr_request,
                    request,
                    profile,
                )
            except Exception as exc:
                return self._failed_result(
                    request,
                    executor_kind="live_ocr",
                    detail=str(exc),
                    warnings=("live_ocr_failed",),
                )
        if request.capability_type == CapabilityType.DESKTOP_INPUT:
            try:
                return await asyncio.to_thread(
                    self._execute_desktop_input,
                    request,
                    should_abort,
                )
            except Exception as exc:
                return self._failed_result(
                    request,
                    executor_kind="live_input",
                    detail=str(exc),
                    warnings=("live_desktop_input_failed",),
                )
        return self.stub_executor.execute(request, decision=decision)

    def _execute_file_operation(
        self,
        request: CapabilityRequest,
        profile: UserSettingsProfile,
    ) -> CapabilityExecutionResult:
        spec = request.file_operation
        if spec is None:
            raise ValueError("File-operation execution requires a file_operation payload.")
        allowlisted_roots = self._allowlisted_roots(profile)
        operation = spec.operation.strip().lower()
        if operation == "read":
            source_path = self._validated_path(spec.source_path, allowlisted_roots=allowlisted_roots)
            if source_path.is_dir():
                entries, truncated = self._list_directory(source_path, recursive=spec.recursive)
                warnings = ("directory_listing_truncated",) if truncated else ()
                return CapabilityExecutionResult(
                    request_id=request.request_id,
                    capability_type=request.capability_type,
                    action_name=request.action_name(),
                    status=CapabilityExecutionStatus.SUCCEEDED,
                    summary=f"Listed {len(entries)} entries from '{source_path.name or source_path}'.",
                    detail=(
                        "Returned a bounded directory listing from the requested allowlisted path."
                    ),
                    executor_kind="live_file",
                    output_ref=str(source_path),
                    warnings=warnings,
                    metadata={
                        "path": str(source_path),
                        "entry_count": len(entries),
                        "entries": list(entries),
                        "recursive": spec.recursive,
                        "truncated": truncated,
                    },
                    completed_at=utc_now(),
                )
            if not source_path.exists():
                raise FileNotFoundError(f"File '{source_path}' does not exist.")
            preview, truncated, file_size = self._read_text_preview(
                source_path,
                max_bytes=self.max_text_file_bytes,
            )
            warnings = ("file_content_truncated",) if truncated else ()
            return CapabilityExecutionResult(
                request_id=request.request_id,
                capability_type=request.capability_type,
                action_name=request.action_name(),
                status=CapabilityExecutionStatus.SUCCEEDED,
                summary=f"Read '{source_path.name}'.",
                detail="Returned a bounded text preview from the allowlisted file path.",
                executor_kind="live_file",
                output_ref=str(source_path),
                warnings=warnings,
                metadata={
                    "path": str(source_path),
                    "content_preview": preview,
                    "truncated": truncated,
                    "file_size_bytes": file_size,
                },
                completed_at=utc_now(),
            )
        if operation == "write":
            target_path = self._validated_path(
                spec.source_path,
                allowlisted_roots=allowlisted_roots,
                allow_missing=True,
            )
            has_content = "content" in request.metadata or "text" in request.metadata
            if not has_content:
                raise ValueError("File write requests must include metadata.content or metadata.text.")
            content = str(request.metadata.get("content", request.metadata.get("text", "")))
            encoded_content = content.encode("utf-8")
            if len(encoded_content) > self.max_text_file_bytes:
                raise ValueError(
                    f"File write payload is too large for bounded execution ({len(encoded_content)} bytes > {self.max_text_file_bytes})."
                )
            if target_path.exists() and target_path.is_dir():
                raise IsADirectoryError(f"Target path '{target_path}' is a directory.")
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(content, encoding="utf-8")
            return CapabilityExecutionResult(
                request_id=request.request_id,
                capability_type=request.capability_type,
                action_name=request.action_name(),
                status=CapabilityExecutionStatus.SUCCEEDED,
                summary=f"Wrote {len(content)} characters to '{target_path.name}'.",
                detail="Completed a bounded file write inside the allowlisted roots.",
                executor_kind="live_file",
                output_ref=str(target_path),
                metadata={
                    "path": str(target_path),
                    "chars_written": len(content),
                    "bytes_written": len(encoded_content),
                },
                completed_at=utc_now(),
            )
        if operation == "copy":
            source_path = self._validated_path(spec.source_path, allowlisted_roots=allowlisted_roots)
            destination_path = self._require_destination_path(spec, allowlisted_roots)
            if source_path.is_dir():
                if not spec.recursive:
                    raise ValueError("Directory copy requests must set recursive=True.")
                if destination_path.exists():
                    raise FileExistsError(f"Destination '{destination_path}' already exists.")
                shutil.copytree(source_path, destination_path)
            else:
                destination_path.parent.mkdir(parents=True, exist_ok=True)
                if destination_path.exists():
                    raise FileExistsError(f"Destination '{destination_path}' already exists.")
                shutil.copy2(source_path, destination_path)
            return CapabilityExecutionResult(
                request_id=request.request_id,
                capability_type=request.capability_type,
                action_name=request.action_name(),
                status=CapabilityExecutionStatus.SUCCEEDED,
                summary=f"Copied '{source_path.name}' to '{destination_path}'.",
                detail="Completed a bounded copy inside the allowlisted roots.",
                executor_kind="live_file",
                output_ref=str(destination_path),
                metadata={
                    "source_path": str(source_path),
                    "destination_path": str(destination_path),
                    "recursive": spec.recursive,
                },
                completed_at=utc_now(),
            )
        if operation == "move":
            source_path = self._validated_path(spec.source_path, allowlisted_roots=allowlisted_roots)
            destination_path = self._require_destination_path(spec, allowlisted_roots)
            if destination_path.exists():
                raise FileExistsError(f"Destination '{destination_path}' already exists.")
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source_path), str(destination_path))
            return CapabilityExecutionResult(
                request_id=request.request_id,
                capability_type=request.capability_type,
                action_name=request.action_name(),
                status=CapabilityExecutionStatus.SUCCEEDED,
                summary=f"Moved '{source_path.name}' to '{destination_path}'.",
                detail="Completed a bounded move inside the allowlisted roots.",
                executor_kind="live_file",
                output_ref=str(destination_path),
                metadata={
                    "source_path": str(source_path),
                    "destination_path": str(destination_path),
                },
                completed_at=utc_now(),
            )
        if operation == "archive":
            source_path = self._validated_path(spec.source_path, allowlisted_roots=allowlisted_roots)
            destination_path = (
                self._require_destination_path(spec, allowlisted_roots)
                if spec.destination_path.strip()
                else self._default_archive_path(source_path, allowlisted_roots)
            )
            if destination_path.exists():
                raise FileExistsError(f"Archive destination '{destination_path}' already exists.")
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source_path), str(destination_path))
            return CapabilityExecutionResult(
                request_id=request.request_id,
                capability_type=request.capability_type,
                action_name=request.action_name(),
                status=CapabilityExecutionStatus.SUCCEEDED,
                summary=f"Archived '{source_path.name}' to '{destination_path}'.",
                detail="Moved the requested file or directory to a bounded archive location.",
                executor_kind="live_file",
                output_ref=str(destination_path),
                metadata={
                    "source_path": str(source_path),
                    "destination_path": str(destination_path),
                },
                completed_at=utc_now(),
            )
        if operation == "delete":
            source_path = self._validated_path(spec.source_path, allowlisted_roots=allowlisted_roots)
            if source_path.is_dir():
                if spec.recursive:
                    shutil.rmtree(source_path)
                else:
                    source_path.rmdir()
            else:
                source_path.unlink()
            return CapabilityExecutionResult(
                request_id=request.request_id,
                capability_type=request.capability_type,
                action_name=request.action_name(),
                status=CapabilityExecutionStatus.SUCCEEDED,
                summary=f"Deleted '{source_path.name}'.",
                detail="Removed the requested file or directory from the allowlisted roots.",
                executor_kind="live_file",
                output_ref=str(source_path),
                metadata={"source_path": str(source_path), "recursive": spec.recursive},
                completed_at=utc_now(),
            )
        raise ValueError(f"Unsupported file operation '{operation}'.")

    def _execute_shell_command(
        self,
        request: CapabilityRequest,
        profile: UserSettingsProfile,
    ) -> CapabilityExecutionResult:
        spec = request.shell_command
        if spec is None:
            raise ValueError("Shell execution requires a shell_command payload.")
        allowlisted_roots = self._allowlisted_roots(profile)
        working_directory = self._validated_path(
            spec.working_directory or ".",
            allowlisted_roots=allowlisted_roots,
        )
        if not working_directory.is_dir():
            raise NotADirectoryError(f"Shell working directory '{working_directory}' is not a directory.")
        resolved_command = shutil.which(spec.command)
        if resolved_command is None:
            raise FileNotFoundError(f"Shell command '{spec.command}' is not available on PATH.")
        timeout_s = max(0.1, float(self.config.backend_runtime.request_timeout_s))
        stdout_preview = ""
        stderr_preview = ""
        stdout_truncated = False
        stderr_truncated = False
        stdout_bytes = 0
        stderr_bytes = 0
        exit_code: int | None = None
        timed_out = False
        stdout_path: Path | None = None
        stderr_path: Path | None = None
        with tempfile.NamedTemporaryFile(delete=False) as stdout_handle, tempfile.NamedTemporaryFile(delete=False) as stderr_handle:
            stdout_path = Path(stdout_handle.name)
            stderr_path = Path(stderr_handle.name)
            try:
                completed = subprocess.run(
                    [resolved_command, *spec.args],
                    cwd=working_directory,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    check=False,
                    timeout=timeout_s,
                    shell=False,
                )
                exit_code = completed.returncode
            except subprocess.TimeoutExpired:
                timed_out = True
            finally:
                stdout_handle.flush()
                stderr_handle.flush()
        try:
            if stdout_path is not None:
                stdout_preview, stdout_truncated, stdout_bytes = self._read_output_preview(
                    stdout_path,
                    max_bytes=self.max_shell_output_bytes,
                )
            if stderr_path is not None:
                stderr_preview, stderr_truncated, stderr_bytes = self._read_output_preview(
                    stderr_path,
                    max_bytes=self.max_shell_output_bytes,
                )
        finally:
            if stdout_path is not None:
                stdout_path.unlink(missing_ok=True)
            if stderr_path is not None:
                stderr_path.unlink(missing_ok=True)
        warnings: list[str] = []
        if stdout_truncated:
            warnings.append("stdout_truncated")
        if stderr_truncated:
            warnings.append("stderr_truncated")
        if timed_out:
            warnings.append("shell_timeout")
            return CapabilityExecutionResult(
                request_id=request.request_id,
                capability_type=request.capability_type,
                action_name=request.action_name(),
                status=CapabilityExecutionStatus.FAILED,
                summary=f"Shell command '{spec.command}' timed out.",
                detail=f"Bounded shell execution hit the {timeout_s:.2f}s timeout and was terminated.",
                executor_kind="live_shell",
                output_ref=str(working_directory),
                warnings=tuple(warnings),
                metadata={
                    "command": [spec.command, *spec.args],
                    "resolved_command": resolved_command,
                    "working_directory": str(working_directory),
                    "timed_out": True,
                    "stdout_preview": stdout_preview,
                    "stderr_preview": stderr_preview,
                    "stdout_bytes": stdout_bytes,
                    "stderr_bytes": stderr_bytes,
                },
                completed_at=utc_now(),
            )
        if exit_code != 0:
            warnings.append("nonzero_exit")
            return CapabilityExecutionResult(
                request_id=request.request_id,
                capability_type=request.capability_type,
                action_name=request.action_name(),
                status=CapabilityExecutionStatus.FAILED,
                summary=f"Shell command '{spec.command}' exited with status {exit_code}.",
                detail="Bounded shell execution completed, but the command reported a nonzero exit status.",
                executor_kind="live_shell",
                output_ref=str(working_directory),
                warnings=tuple(warnings),
                metadata={
                    "command": [spec.command, *spec.args],
                    "resolved_command": resolved_command,
                    "working_directory": str(working_directory),
                    "exit_code": exit_code,
                    "stdout_preview": stdout_preview,
                    "stderr_preview": stderr_preview,
                    "stdout_bytes": stdout_bytes,
                    "stderr_bytes": stderr_bytes,
                },
                completed_at=utc_now(),
            )
        return CapabilityExecutionResult(
            request_id=request.request_id,
            capability_type=request.capability_type,
            action_name=request.action_name(),
            status=CapabilityExecutionStatus.SUCCEEDED,
            summary=f"Shell command '{spec.command}' completed successfully.",
            detail="Bounded shell execution completed inside the allowlisted working directory.",
            executor_kind="live_shell",
            output_ref=str(working_directory),
            warnings=tuple(warnings),
            metadata={
                "command": [spec.command, *spec.args],
                "resolved_command": resolved_command,
                "working_directory": str(working_directory),
                "exit_code": exit_code,
                "stdout_preview": stdout_preview,
                "stderr_preview": stderr_preview,
                "stdout_bytes": stdout_bytes,
                "stderr_bytes": stderr_bytes,
            },
            completed_at=utc_now(),
        )

    def _execute_browser_action(
        self,
        request: CapabilityRequest,
    ) -> CapabilityExecutionResult:
        spec = request.browser_action
        if spec is None:
            raise ValueError("Browser execution requires a browser_action payload.")
        action = spec.action.strip().lower()
        domain = spec.domain.strip() or urlparse(spec.url).netloc.split(":")[0]
        if action == "read":
            return self._execute_browser_read(request, spec=spec, domain=domain)
        if action == "navigate":
            return self._execute_browser_navigate(request, spec=spec, domain=domain)
        if action in {"click", "type", "download"}:
            return CapabilityExecutionResult(
                request_id=request.request_id,
                capability_type=request.capability_type,
                action_name=request.action_name(),
                status=CapabilityExecutionStatus.FAILED,
                summary=f"Browser action '{action}' is deferred.",
                detail=(
                    "Interactive browser actions remain deferred until the tighter desktop-input tier lands."
                ),
                executor_kind="live_browser",
                output_ref=spec.url or domain,
                warnings=("browser_interaction_deferred",),
                metadata={
                    "action": action,
                    "url": spec.url,
                    "domain": domain,
                },
                completed_at=utc_now(),
            )
        raise ValueError(f"Unsupported browser action '{action}'.")

    def _execute_browser_read(
        self,
        request: CapabilityRequest,
        *,
        spec: object,
        domain: str,
    ) -> CapabilityExecutionResult:
        request_url = str(getattr(spec, "url", "")).strip()
        if not request_url:
            raise ValueError("Browser read requests must include a URL.")
        timeout_s = min(_WINDOW_MATCH_TIMEOUT_S, max(0.25, float(self.config.backend_runtime.request_timeout_s)))
        response_headers: dict[str, str] = {}
        status_code = 0
        with urllib_request.urlopen(
            urllib_request.Request(
                request_url,
                headers={"User-Agent": self.config.web.user_agent},
            ),
            timeout=timeout_s,
        ) as response:
            status_code = getattr(response, "status", 200) or 200
            response_headers = {key.lower(): value for key, value in response.headers.items()}
            preview_bytes = response.read(self.max_shell_output_bytes + 1)
        truncated = len(preview_bytes) > self.max_shell_output_bytes
        charset = "utf-8"
        content_type = response_headers.get("content-type", "")
        match = re.search(r"charset=([a-zA-Z0-9._-]+)", content_type)
        if match:
            charset = match.group(1)
        preview = preview_bytes[: self.max_shell_output_bytes].decode(charset, errors="replace")
        page_title = self._extract_html_title(preview)
        expected_title = str(
            request.metadata.get("expected_title", request.metadata.get("expected_window_title", ""))
        ).strip()
        if expected_title and expected_title.lower() not in page_title.lower():
            raise RuntimeError(
                f"Fetched page title '{page_title or '(missing title)'}' did not match expected title '{expected_title}'."
            )
        warnings = ("browser_read_truncated",) if truncated else ()
        return CapabilityExecutionResult(
            request_id=request.request_id,
            capability_type=request.capability_type,
            action_name=request.action_name(),
            status=CapabilityExecutionStatus.SUCCEEDED,
            summary=f"Read allowlisted browser target '{domain}'.",
            detail="Fetched a bounded browser read preview and validated the returned title when requested.",
            executor_kind="live_browser",
            output_ref=request_url,
            warnings=warnings,
            metadata={
                "url": request_url,
                "domain": domain,
                "status_code": status_code,
                "content_type": content_type,
                "page_title": page_title,
                "content_preview": preview,
                "truncated": truncated,
            },
            completed_at=utc_now(),
        )

    def _execute_browser_navigate(
        self,
        request: CapabilityRequest,
        *,
        spec: object,
        domain: str,
    ) -> CapabilityExecutionResult:
        request_url = str(getattr(spec, "url", "")).strip()
        if not request_url:
            raise ValueError("Browser navigate requests must include a URL.")
        expected_title = str(
            request.metadata.get("expected_title", request.metadata.get("expected_window_title", ""))
        ).strip()
        self._open_browser_url(request_url)
        matched_window = self._wait_for_window_match(
            process_names=self._browser_process_names(
                request.metadata.get("browser_process_names", ()),
            ),
            title_contains=expected_title or domain,
        )
        if matched_window is None:
            raise RuntimeError(
                f"No visible browser window matched '{expected_title or domain}' after navigation."
            )
        self._focus_window(matched_window.hwnd)
        foreground_title = self._foreground_window_title()
        validation_token = (expected_title or matched_window.title or domain).strip().lower()
        if validation_token and validation_token not in foreground_title.lower():
            raise RuntimeError(
                f"Foreground window title '{foreground_title or '(missing)'}' did not match '{validation_token}'."
            )
        return CapabilityExecutionResult(
            request_id=request.request_id,
            capability_type=request.capability_type,
            action_name=request.action_name(),
            status=CapabilityExecutionStatus.SUCCEEDED,
            summary=f"Navigated the local browser to '{domain}'.",
            detail="Opened an allowlisted URL and validated a visible browser target window.",
            executor_kind="live_browser",
            output_ref=request_url,
            metadata={
                "url": request_url,
                "domain": domain,
                "matched_window_title": matched_window.title,
                "matched_process_name": matched_window.process_name,
                "foreground_title": foreground_title,
            },
            completed_at=utc_now(),
        )

    def _execute_app_focus(
        self,
        request: CapabilityRequest,
    ) -> CapabilityExecutionResult:
        spec = request.app_focus
        if spec is None:
            raise ValueError("App/window execution requires an app_focus payload.")
        matched_window = self._wait_for_window_match(
            app_name=spec.app_name,
            title_contains=spec.window_title,
        )
        if matched_window is None:
            raise RuntimeError(
                f"No visible window matched app '{spec.app_name}'"
                + (f" and title '{spec.window_title}'." if spec.window_title.strip() else ".")
            )
        self._focus_window(matched_window.hwnd)
        foreground_title = self._foreground_window_title()
        validation_token = (
            spec.window_title.strip().lower()
            or matched_window.title.strip().lower()
        )
        if spec.require_visible_match and validation_token and validation_token not in foreground_title.lower():
            raise RuntimeError(
                f"Foreground window title '{foreground_title or '(missing)'}' did not match '{validation_token}'."
            )
        return CapabilityExecutionResult(
            request_id=request.request_id,
            capability_type=request.capability_type,
            action_name=request.action_name(),
            status=CapabilityExecutionStatus.SUCCEEDED,
            summary=f"Focused '{matched_window.title or spec.app_name}'.",
            detail="Focused a visible allowlisted application window and validated the foreground title.",
            executor_kind="live_window",
            output_ref=matched_window.title or spec.app_name,
            metadata={
                "app_name": spec.app_name,
                "matched_window_title": matched_window.title,
                "matched_process_name": matched_window.process_name,
                "matched_pid": matched_window.pid,
                "foreground_title": foreground_title,
            },
            completed_at=utc_now(),
        )

    def _execute_screenshot(
        self,
        request: CapabilityRequest,
        profile: UserSettingsProfile,
    ) -> CapabilityExecutionResult:
        spec = request.screenshot
        if spec is None:
            raise ValueError("Screenshot execution requires a screenshot payload.")
        allowlisted_roots = self._allowlisted_roots(profile)
        save_path = self._validated_path(
            spec.save_path,
            allowlisted_roots=allowlisted_roots,
            allow_missing=True,
        )
        if save_path.exists() and save_path.is_dir():
            raise IsADirectoryError(f"Screenshot output path '{save_path}' is a directory.")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        region = self._parse_capture_region(spec.region, full_token="full_screen")
        image_format = self._image_format_for_path(save_path)
        self._capture_screenshot_file(save_path, region=region, image_format=image_format)
        if not save_path.exists():
            raise FileNotFoundError(f"Screenshot output '{save_path}' was not created.")
        file_size_bytes = save_path.stat().st_size
        return CapabilityExecutionResult(
            request_id=request.request_id,
            capability_type=request.capability_type,
            action_name=request.action_name(),
            status=CapabilityExecutionStatus.SUCCEEDED,
            summary=f"Captured a screenshot to '{save_path.name}'.",
            detail="Saved a bounded on-demand screenshot inside the allowlisted roots.",
            executor_kind="live_screenshot",
            output_ref=str(save_path),
            metadata={
                "save_path": str(save_path),
                "region": spec.region,
                "image_format": image_format.lower(),
                "file_size_bytes": file_size_bytes,
                "capture_bounds": (
                    {
                        "left": region[0],
                        "top": region[1],
                        "width": region[2],
                        "height": region[3],
                    }
                    if region is not None
                    else "virtual_screen"
                ),
            },
            completed_at=utc_now(),
        )

    def _execute_ocr_request(
        self,
        request: CapabilityRequest,
        profile: UserSettingsProfile,
    ) -> CapabilityExecutionResult:
        spec = request.ocr_request
        if spec is None:
            raise ValueError("OCR execution requires an ocr_request payload.")
        allowlisted_roots = self._allowlisted_roots(profile)
        source_path = self._validated_path(
            spec.source_image_path,
            allowlisted_roots=allowlisted_roots,
        )
        languages = tuple(language for language in spec.languages if language.strip())
        bounded_text, backend_name, warnings, text_length, truncated = self.extract_bounded_ocr_text(
            source_path,
            region=spec.region or "full_image",
            languages=languages,
        )
        return CapabilityExecutionResult(
            request_id=request.request_id,
            capability_type=request.capability_type,
            action_name=request.action_name(),
            status=CapabilityExecutionStatus.SUCCEEDED,
            summary=f"OCR completed for '{source_path.name}'.",
            detail="Ran CPU-first OCR against the bounded local image input.",
            executor_kind="live_ocr",
            output_ref=str(source_path),
            warnings=tuple(warnings),
            metadata={
                "source_image_path": str(source_path),
                "ocr_backend": backend_name,
                "region": spec.region,
                "languages": languages,
                "recognized_text": bounded_text,
                "text_length": text_length,
                "truncated": truncated,
            },
            completed_at=utc_now(),
        )

    def capture_observation_frame(
        self,
        destination_path: Path,
        *,
        region: str,
        max_width: int,
        max_height: int,
        image_format: str = "Png",
    ) -> None:
        """Capture one bounded observation frame into the requested destination."""
        parsed_region = self._parse_capture_region(region, full_token="full_screen")
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        self._capture_screenshot_file(
            destination_path,
            region=parsed_region,
            image_format=image_format,
            resize_to=(max_width, max_height),
        )

    def extract_bounded_ocr_text(
        self,
        source_path: Path,
        *,
        region: str = "full_image",
        languages: tuple[str, ...] = (),
    ) -> tuple[str, str, tuple[str, ...], int, bool]:
        """Run bounded CPU-first OCR and return text, backend, warnings, length, and truncation state."""
        crop_region = self._parse_capture_region(region, full_token="full_image")
        crop_path: Path | None = None
        image_path = source_path
        try:
            if crop_region is not None:
                image_path, crop_path = self._crop_image_for_ocr(source_path, crop_region)
            recognized_text, backend_name = self._extract_ocr_text(
                image_path,
                languages=tuple(language for language in languages if language.strip()),
            )
        finally:
            if crop_path is not None:
                crop_path.unlink(missing_ok=True)
        normalized_text = recognized_text.strip()
        text_length = len(normalized_text)
        truncated = text_length > _MAX_OCR_TEXT_CHARS
        bounded_text = normalized_text[:_MAX_OCR_TEXT_CHARS]
        warnings: list[str] = []
        if truncated:
            warnings.append("ocr_text_truncated")
        if not bounded_text:
            warnings.append("ocr_no_text_detected")
        return bounded_text, backend_name, tuple(warnings), text_length, truncated

    def capture_continuous_frame(
        self,
        destination_path: Path,
        *,
        region: str,
        max_width: int,
        max_height: int,
    ) -> None:
        """Capture one bounded continuous-observation frame into the requested destination."""
        self.capture_observation_frame(
            destination_path,
            region=region,
            max_width=max_width,
            max_height=max_height,
            image_format="Jpeg",
        )

    def _execute_desktop_input(
        self,
        request: CapabilityRequest,
        should_abort: Callable[[], str | None] | None = None,
    ) -> CapabilityExecutionResult:
        spec = request.desktop_input
        if spec is None:
            raise ValueError("Desktop-input execution requires a desktop_input payload.")
        if os.name != "nt":
            raise RuntimeError("Desktop-input execution currently requires Windows.")
        target_name = spec.target.strip()
        if not target_name:
            raise ValueError("Desktop-input execution requires a non-empty target app name.")
        expected_title = str(request.metadata.get("expected_window_title", "")).strip()
        matched_window = self._wait_for_window_match(
            app_name=target_name,
            title_contains=expected_title,
        )
        if matched_window is None:
            return self._desktop_input_failure(
                request,
                detail=(
                    f"No visible window matched target '{target_name}'"
                    + (f" and title '{expected_title}'." if expected_title else ".")
                ),
                warning="target_window_not_found",
            )
        self._focus_window(matched_window.hwnd)
        validation_token = (expected_title or matched_window.title or target_name).strip().lower()
        if abort_reason := self._desktop_input_guard(
            validation_token=validation_token,
            should_abort=should_abort,
        ):
            return self._desktop_input_failure(
                request,
                detail=self._desktop_input_abort_detail(abort_reason, validation_token=validation_token),
                warning=abort_reason,
            )
        action = spec.action.strip().lower()
        metadata: dict[str, object] = {
            "target": target_name,
            "matched_window_title": matched_window.title,
            "matched_process_name": matched_window.process_name,
            "matched_pid": matched_window.pid,
        }
        if action == "type_text":
            if not spec.text:
                raise ValueError("DesktopInputSpec.text must not be empty for type_text actions.")
            for character in spec.text:
                if abort_reason := self._desktop_input_guard(
                    validation_token=validation_token,
                    should_abort=should_abort,
                ):
                    return self._desktop_input_failure(
                        request,
                        detail=self._desktop_input_abort_detail(abort_reason, validation_token=validation_token),
                        warning=abort_reason,
                    )
                self._send_text_input(character)
            foreground_title = self._foreground_window_title()
            metadata["foreground_title"] = foreground_title
            return CapabilityExecutionResult(
                request_id=request.request_id,
                capability_type=request.capability_type,
                action_name=request.action_name(),
                status=CapabilityExecutionStatus.SUCCEEDED,
                summary=f"Typed text into '{matched_window.title or target_name}'.",
                detail="Typed bounded text into the validated foreground target window.",
                executor_kind="live_input",
                output_ref=matched_window.title or target_name,
                metadata={**metadata, "chars_typed": len(spec.text)},
                completed_at=utc_now(),
            )
        if action == "press_keys":
            if not spec.keys:
                raise ValueError("DesktopInputSpec.keys must not be empty for press_keys actions.")
            if abort_reason := self._desktop_input_guard(
                validation_token=validation_token,
                should_abort=should_abort,
            ):
                return self._desktop_input_failure(
                    request,
                    detail=self._desktop_input_abort_detail(abort_reason, validation_token=validation_token),
                    warning=abort_reason,
                )
            self._send_key_chord(spec.keys)
            foreground_title = self._foreground_window_title()
            metadata["foreground_title"] = foreground_title
            return CapabilityExecutionResult(
                request_id=request.request_id,
                capability_type=request.capability_type,
                action_name=request.action_name(),
                status=CapabilityExecutionStatus.SUCCEEDED,
                summary=f"Pressed keys in '{matched_window.title or target_name}'.",
                detail="Sent one bounded key chord to the validated foreground target window.",
                executor_kind="live_input",
                output_ref=matched_window.title or target_name,
                metadata={**metadata, "keys": list(spec.keys)},
                completed_at=utc_now(),
            )
        if action == "mouse_move":
            if spec.x is None or spec.y is None:
                raise ValueError("Desktop mouse_move actions require both x and y coordinates.")
            if abort_reason := self._perform_mouse_move(
                spec.x,
                spec.y,
                validation_token=validation_token,
                should_abort=should_abort,
            ):
                return self._desktop_input_failure(
                    request,
                    detail=self._desktop_input_abort_detail(abort_reason, validation_token=validation_token),
                    warning=abort_reason,
                )
            foreground_title = self._foreground_window_title()
            metadata["foreground_title"] = foreground_title
            return CapabilityExecutionResult(
                request_id=request.request_id,
                capability_type=request.capability_type,
                action_name=request.action_name(),
                status=CapabilityExecutionStatus.SUCCEEDED,
                summary=f"Moved the mouse for '{matched_window.title or target_name}'.",
                detail="Moved the cursor in bounded steps while validating the active target window.",
                executor_kind="live_input",
                output_ref=matched_window.title or target_name,
                metadata={**metadata, "x": spec.x, "y": spec.y},
                completed_at=utc_now(),
            )
        if action == "mouse_click":
            click_count = max(1, min(int(request.metadata.get("clicks", 1) or 1), _MAX_DESKTOP_CLICK_COUNT))
            button = str(request.metadata.get("button", "left")).strip().lower() or "left"
            if spec.x is not None and spec.y is not None:
                if abort_reason := self._perform_mouse_move(
                    spec.x,
                    spec.y,
                    validation_token=validation_token,
                    should_abort=should_abort,
                ):
                    return self._desktop_input_failure(
                        request,
                        detail=self._desktop_input_abort_detail(abort_reason, validation_token=validation_token),
                        warning=abort_reason,
                    )
            for _ in range(click_count):
                if abort_reason := self._desktop_input_guard(
                    validation_token=validation_token,
                    should_abort=should_abort,
                ):
                    return self._desktop_input_failure(
                        request,
                        detail=self._desktop_input_abort_detail(abort_reason, validation_token=validation_token),
                        warning=abort_reason,
                    )
                self._send_mouse_click(button)
            foreground_title = self._foreground_window_title()
            metadata["foreground_title"] = foreground_title
            return CapabilityExecutionResult(
                request_id=request.request_id,
                capability_type=request.capability_type,
                action_name=request.action_name(),
                status=CapabilityExecutionStatus.SUCCEEDED,
                summary=f"Clicked inside '{matched_window.title or target_name}'.",
                detail="Executed a bounded mouse click sequence against the validated foreground target window.",
                executor_kind="live_input",
                output_ref=matched_window.title or target_name,
                metadata={**metadata, "button": button, "clicks": click_count, "x": spec.x, "y": spec.y},
                completed_at=utc_now(),
            )
        raise ValueError(f"Unsupported desktop input action '{action}'.")

    def _allowlisted_roots(self, profile: UserSettingsProfile) -> tuple[Path, ...]:
        roots = []
        for item in profile.desktop.get("allowlisted_roots", ()):
            raw_root = str(item).strip()
            if not raw_root:
                continue
            roots.append(self._resolve_root(raw_root))
        return tuple(roots)

    def _validated_path(
        self,
        raw_path: str,
        *,
        allowlisted_roots: tuple[Path, ...],
        allow_missing: bool = False,
    ) -> Path:
        candidate = self._resolve_root(raw_path)
        if not any(candidate.is_relative_to(root) for root in allowlisted_roots):
            raise PermissionError(f"Path '{candidate}' is outside the allowlisted roots.")
        if not allow_missing and not candidate.exists():
            raise FileNotFoundError(f"Path '{candidate}' does not exist.")
        return candidate

    def _require_destination_path(
        self,
        spec: object,
        allowlisted_roots: tuple[Path, ...],
    ) -> Path:
        destination_path = str(getattr(spec, "destination_path", "")).strip()
        if not destination_path:
            raise ValueError("This file operation requires destination_path.")
        return self._validated_path(
            destination_path,
            allowlisted_roots=allowlisted_roots,
            allow_missing=True,
        )

    def _default_archive_path(
        self,
        source_path: Path,
        allowlisted_roots: tuple[Path, ...],
    ) -> Path:
        archive_root = source_path.parent / ".quester_archive"
        archive_name = f"{source_path.name}.{utc_now().strftime('%Y%m%dT%H%M%SZ')}"
        return self._validated_path(
            str(archive_root / archive_name),
            allowlisted_roots=allowlisted_roots,
            allow_missing=True,
        )

    def _list_directory(
        self,
        directory: Path,
        *,
        recursive: bool,
    ) -> tuple[tuple[str, ...], bool]:
        iterator = directory.rglob("*") if recursive else directory.iterdir()
        entries: list[str] = []
        truncated = False
        for index, entry in enumerate(iterator):
            if index >= self.max_directory_listing_entries:
                truncated = True
                break
            try:
                entries.append(str(entry.relative_to(directory)))
            except ValueError:
                entries.append(str(entry))
        return tuple(entries), truncated

    @staticmethod
    def _parse_capture_region(
        region: str,
        *,
        full_token: str,
    ) -> tuple[int, int, int, int] | None:
        normalized = region.strip().lower()
        if not normalized or normalized == full_token:
            return None
        parts = [part.strip() for part in region.split(",")]
        if len(parts) != 4:
            raise ValueError(
                f"Region '{region}' must be '{full_token}' or 'left,top,width,height'."
            )
        left, top, width, height = (int(part) for part in parts)
        if width <= 0 or height <= 0:
            raise ValueError("Region width and height must be positive integers.")
        return left, top, width, height

    @staticmethod
    def _image_format_for_path(path: Path) -> str:
        suffix = path.suffix.strip().lower()
        if not suffix or suffix == ".png":
            return "Png"
        if suffix == ".bmp":
            return "Bmp"
        if suffix in {".jpg", ".jpeg"}:
            return "Jpeg"
        raise ValueError("Screenshot outputs must use .png, .bmp, .jpg, or .jpeg.")

    def _capture_screenshot_file(
        self,
        destination_path: Path,
        *,
        region: tuple[int, int, int, int] | None,
        image_format: str,
        resize_to: tuple[int, int] | None = None,
    ) -> None:
        if os.name != "nt":
            raise RuntimeError("Screenshot capture currently requires Windows.")
        region_spec = (
            ""
            if region is None
            else ",".join(str(value) for value in region)
        )
        resize_width = 0 if resize_to is None else max(0, int(resize_to[0]))
        resize_height = 0 if resize_to is None else max(0, int(resize_to[1]))
        script = """
& {
    param(
        [string]$SavePath,
        [string]$RegionSpec,
        [string]$ImageFormat,
        [int]$ResizeWidth,
        [int]$ResizeHeight
    )
    $ErrorActionPreference = 'Stop'
    Add-Type -AssemblyName System.Drawing
    Add-Type -AssemblyName System.Windows.Forms
    if ([string]::IsNullOrWhiteSpace($RegionSpec)) {
        $bounds = [System.Windows.Forms.SystemInformation]::VirtualScreen
        $left = [int]$bounds.Left
        $top = [int]$bounds.Top
        $width = [int]$bounds.Width
        $height = [int]$bounds.Height
    }
    else {
        $parts = $RegionSpec.Split(',')
        if ($parts.Length -ne 4) {
            throw "RegionSpec must contain four comma-separated integers."
        }
        $left = [int]$parts[0]
        $top = [int]$parts[1]
        $width = [int]$parts[2]
        $height = [int]$parts[3]
    }
    if ($width -le 0 -or $height -le 0) {
        throw "Screenshot bounds must have positive width and height."
    }
    $bitmap = New-Object System.Drawing.Bitmap $width, $height
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    try {
        $graphics.CopyFromScreen($left, $top, 0, 0, $bitmap.Size)
        $output = $bitmap
        if ($ResizeWidth -gt 0 -and $ResizeHeight -gt 0 -and ($bitmap.Width -gt $ResizeWidth -or $bitmap.Height -gt $ResizeHeight)) {
            $scale = [Math]::Min($ResizeWidth / [double]$bitmap.Width, $ResizeHeight / [double]$bitmap.Height)
            $scaledWidth = [Math]::Max(1, [int][Math]::Round($bitmap.Width * $scale))
            $scaledHeight = [Math]::Max(1, [int][Math]::Round($bitmap.Height * $scale))
            $resized = New-Object System.Drawing.Bitmap $scaledWidth, $scaledHeight
            $resizeGraphics = [System.Drawing.Graphics]::FromImage($resized)
            try {
                $resizeGraphics.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
                $resizeGraphics.DrawImage($bitmap, 0, 0, $scaledWidth, $scaledHeight)
            }
            finally {
                $resizeGraphics.Dispose()
            }
            $output = $resized
        }
        try {
            $format = [System.Drawing.Imaging.ImageFormat]::$ImageFormat
            $output.Save($SavePath, $format)
        }
        finally {
            if ($output -ne $bitmap) {
                $output.Dispose()
            }
        }
    }
    finally {
        $graphics.Dispose()
        $bitmap.Dispose()
    }
}
"""
        self._run_powershell_command(
            script,
            str(destination_path),
            region_spec,
            image_format,
            str(resize_width),
            str(resize_height),
        )

    def _crop_image_for_ocr(
        self,
        source_path: Path,
        region: tuple[int, int, int, int],
    ) -> tuple[Path, Path]:
        left, top, width, height = region
        if left < 0 or top < 0:
            raise ValueError("OCR crop regions must start within the source image bounds.")
        fd, raw_temp_path = tempfile.mkstemp(
            prefix=f"{source_path.stem}_ocr_",
            suffix=".png",
            dir=str(source_path.parent),
        )
        os.close(fd)
        destination_path = Path(raw_temp_path)
        script = """
& {
    param(
        [string]$SourcePath,
        [string]$DestinationPath,
        [int]$Left,
        [int]$Top,
        [int]$Width,
        [int]$Height
    )
    $ErrorActionPreference = 'Stop'
    Add-Type -AssemblyName System.Drawing
    $bitmap = [System.Drawing.Bitmap]::FromFile($SourcePath)
    try {
        if ($Left -lt 0 -or $Top -lt 0 -or $Width -le 0 -or $Height -le 0) {
            throw "OCR crop bounds must be positive and within the image."
        }
        if (($Left + $Width) -gt $bitmap.Width -or ($Top + $Height) -gt $bitmap.Height) {
            throw "OCR crop bounds exceed the source image."
        }
        $rect = New-Object System.Drawing.Rectangle($Left, $Top, $Width, $Height)
        $cropped = $bitmap.Clone($rect, $bitmap.PixelFormat)
        try {
            $cropped.Save($DestinationPath, [System.Drawing.Imaging.ImageFormat]::Png)
        }
        finally {
            $cropped.Dispose()
        }
    }
    finally {
        $bitmap.Dispose()
    }
}
"""
        try:
            self._run_powershell_command(
                script,
                str(source_path),
                str(destination_path),
                str(left),
                str(top),
                str(width),
                str(height),
            )
        except Exception:
            destination_path.unlink(missing_ok=True)
            raise
        return destination_path, destination_path

    def _extract_ocr_text(
        self,
        image_path: Path,
        *,
        languages: tuple[str, ...],
    ) -> tuple[str, str]:
        backend_errors: list[str] = []
        if os.name == "nt":
            try:
                return self._ocr_with_windows_runtime(image_path, languages=languages), "windows_ocr"
            except Exception as exc:
                backend_errors.append(f"windows_ocr: {exc}")
        tesseract_path = shutil.which("tesseract")
        if tesseract_path:
            try:
                return self._ocr_with_tesseract(image_path, tesseract_path=tesseract_path, languages=languages), "tesseract"
            except Exception as exc:
                backend_errors.append(f"tesseract: {exc}")
        raise RuntimeError(
            "No local CPU OCR backend is available. "
            + ("; ".join(backend_errors) if backend_errors else "Windows OCR or tesseract is required.")
        )

    def _ocr_with_windows_runtime(
        self,
        image_path: Path,
        *,
        languages: tuple[str, ...],
    ) -> str:
        language_hint = next((language.strip() for language in languages if language.strip()), "")
        script = """
& {
    param(
        [string]$ImagePath,
        [string]$LanguageTag
    )
    $ErrorActionPreference = 'Stop'
    Add-Type -AssemblyName System.Runtime.WindowsRuntime
    $null = [Windows.Storage.StorageFile, Windows.Storage, ContentType = WindowsRuntime]
    $null = [Windows.Graphics.Imaging.BitmapDecoder, Windows.Graphics.Imaging, ContentType = WindowsRuntime]
    $null = [Windows.Media.Ocr.OcrEngine, Windows.Media.Ocr, ContentType = WindowsRuntime]
    function Await([object]$Operation) {
        $task = [System.WindowsRuntimeSystemExtensions]::AsTask($Operation)
        $task.Wait()
        if ($task.Exception) {
            throw $task.Exception
        }
        return $task.Result
    }
    $file = Await([Windows.Storage.StorageFile]::GetFileFromPathAsync($ImagePath))
    $stream = Await($file.OpenAsync([Windows.Storage.FileAccessMode]::Read))
    try {
        $decoder = Await([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream))
        $bitmap = Await($decoder.GetSoftwareBitmapAsync())
    }
    finally {
        $stream.Dispose()
    }
    $engine = $null
    if (-not [string]::IsNullOrWhiteSpace($LanguageTag)) {
        try {
            $language = New-Object Windows.Globalization.Language($LanguageTag)
            $engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage($language)
        }
        catch {
            $engine = $null
        }
    }
    if (-not $engine) {
        $engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages()
    }
    if (-not $engine) {
        throw "Windows OCR engine is unavailable."
    }
    $result = Await($engine.RecognizeAsync($bitmap))
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    Write-Output $result.Text
}
"""
        return self._run_powershell_command(
            script,
            str(image_path),
            language_hint,
        )

    def _ocr_with_tesseract(
        self,
        image_path: Path,
        *,
        tesseract_path: str,
        languages: tuple[str, ...],
    ) -> str:
        language_arg = "+".join(language.strip() for language in languages if language.strip()) or "eng"
        timeout_s = max(0.5, float(self.config.backend_runtime.request_timeout_s))
        completed = subprocess.run(
            [tesseract_path, str(image_path), "stdout", "-l", language_arg],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=timeout_s,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or "Tesseract OCR returned a nonzero exit status.")
        return completed.stdout

    def _run_powershell_command(
        self,
        script: str,
        *args: str,
    ) -> str:
        executable = shutil.which("powershell") or shutil.which("pwsh")
        if executable is None:
            raise RuntimeError("PowerShell is not available on PATH.")
        timeout_s = max(0.5, float(self.config.backend_runtime.request_timeout_s))
        completed = subprocess.run(
            [executable, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script, *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=timeout_s,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "PowerShell command failed.")
        return completed.stdout.strip()

    def _desktop_input_guard(
        self,
        *,
        validation_token: str,
        should_abort: Callable[[], str | None] | None,
    ) -> str | None:
        if should_abort is not None:
            abort_reason = should_abort()
            if abort_reason:
                return abort_reason
        foreground_title = self._foreground_window_title().strip().lower()
        if validation_token and validation_token not in foreground_title:
            return "foreground_target_mismatch"
        return None

    def _perform_mouse_move(
        self,
        x: int,
        y: int,
        *,
        validation_token: str,
        should_abort: Callable[[], str | None] | None,
    ) -> str | None:
        current_x, current_y = self._cursor_position()
        steps = max(1, min(_MAX_MOUSE_MOVE_STEPS, max(abs(x - current_x), abs(y - current_y)) // 25 + 1))
        for step in range(1, steps + 1):
            if abort_reason := self._desktop_input_guard(
                validation_token=validation_token,
                should_abort=should_abort,
            ):
                return abort_reason
            next_x = round(current_x + ((x - current_x) * step / steps))
            next_y = round(current_y + ((y - current_y) * step / steps))
            self._set_cursor_position(next_x, next_y)
            if steps > 1:
                time.sleep(0.01)
        return None

    @staticmethod
    def _desktop_input_abort_detail(
        abort_reason: str,
        *,
        validation_token: str,
    ) -> str:
        if abort_reason == "emergency_stop_requested":
            return "The local task emergency stop was requested while desktop input was running."
        if abort_reason == "foreground_target_mismatch":
            return (
                "The foreground target changed unexpectedly during desktop input "
                f"(expected title token '{validation_token or '(none)'}')."
            )
        return f"Desktop input aborted because '{abort_reason}' was raised."

    def _desktop_input_failure(
        self,
        request: CapabilityRequest,
        *,
        detail: str,
        warning: str,
    ) -> CapabilityExecutionResult:
        return CapabilityExecutionResult(
            request_id=request.request_id,
            capability_type=request.capability_type,
            action_name=request.action_name(),
            status=CapabilityExecutionStatus.FAILED,
            summary=f"Desktop input '{request.action_name()}' did not complete.",
            detail=detail,
            executor_kind="live_input",
            output_ref=request.target_summary(),
            warnings=(warning,),
            metadata={"target": request.target_summary(), "summary": request.summary},
            completed_at=utc_now(),
        )

    @staticmethod
    def _browser_process_names(extra_process_names: object = ()) -> tuple[str, ...]:
        names = {"brave", "chrome", "firefox", "iexplore", "msedge", "opera"}
        for item in extra_process_names if isinstance(extra_process_names, (list, tuple, set)) else ():
            name = str(item).strip().lower()
            if name:
                names.add(name.removesuffix(".exe"))
        return tuple(sorted(names))

    @staticmethod
    def _extract_html_title(content: str) -> str:
        title_match = re.search(r"<title[^>]*>(.*?)</title>", content, flags=re.IGNORECASE | re.DOTALL)
        if title_match is None:
            return ""
        return re.sub(r"\s+", " ", title_match.group(1)).strip()

    def _open_browser_url(self, url: str) -> None:
        if os.name == "nt" and hasattr(os, "startfile"):
            os.startfile(url)  # type: ignore[attr-defined]
            return
        if not webbrowser.open(url, new=2):
            raise RuntimeError(f"Could not open browser URL '{url}'.")

    def _wait_for_window_match(
        self,
        *,
        app_name: str = "",
        process_names: tuple[str, ...] = (),
        title_contains: str = "",
        timeout_s: float = _WINDOW_MATCH_TIMEOUT_S,
    ) -> WindowSnapshot | None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() <= deadline:
            matched_window = self._match_window(
                app_name=app_name,
                process_names=process_names,
                title_contains=title_contains,
            )
            if matched_window is not None:
                return matched_window
            time.sleep(_WINDOW_MATCH_POLL_S)
        return None

    def _match_window(
        self,
        *,
        app_name: str = "",
        process_names: tuple[str, ...] = (),
        title_contains: str = "",
    ) -> WindowSnapshot | None:
        expected_app = app_name.strip().lower().removesuffix(".exe")
        allowed_processes = {item.strip().lower().removesuffix(".exe") for item in process_names if str(item).strip()}
        expected_title = title_contains.strip().lower()
        for window in self._enumerate_visible_windows():
            process_name = window.process_name.strip().lower().removesuffix(".exe")
            if expected_app and process_name != expected_app:
                continue
            if allowed_processes and process_name not in allowed_processes:
                continue
            if expected_title and expected_title not in window.title.lower():
                continue
            return window
        return None

    def _enumerate_visible_windows(self) -> tuple[WindowSnapshot, ...]:
        if os.name != "nt":
            return ()
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        windows: list[WindowSnapshot] = []
        enum_windows_proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

        def callback(hwnd: int, _lparam: int) -> bool:
            if not user32.IsWindowVisible(hwnd):
                return True
            title_length = user32.GetWindowTextLengthW(hwnd)
            if title_length <= 0:
                return True
            buffer = ctypes.create_unicode_buffer(title_length + 1)
            user32.GetWindowTextW(hwnd, buffer, title_length + 1)
            title = buffer.value.strip()
            if not title:
                return True
            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            windows.append(
                WindowSnapshot(
                    hwnd=int(hwnd),
                    title=title,
                    process_name=self._process_name_for_pid(int(pid.value)),
                    pid=int(pid.value),
                )
            )
            return True

        user32.EnumWindows(enum_windows_proc(callback), 0)
        return tuple(windows)

    def _process_name_for_pid(self, pid: int) -> str:
        if os.name != "nt":
            return ""
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        process_handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not process_handle:
            return ""
        try:
            buffer_length = wintypes.DWORD(260)
            buffer = ctypes.create_unicode_buffer(buffer_length.value)
            if not kernel32.QueryFullProcessImageNameW(
                process_handle,
                0,
                buffer,
                ctypes.byref(buffer_length),
            ):
                return ""
            return Path(buffer.value).stem.lower()
        finally:
            kernel32.CloseHandle(process_handle)

    def _focus_window(self, hwnd: int) -> bool:
        if os.name != "nt":
            return False
        import ctypes

        user32 = ctypes.windll.user32
        SW_RESTORE = 9
        user32.ShowWindow(hwnd, SW_RESTORE)
        user32.BringWindowToTop(hwnd)
        return bool(user32.SetForegroundWindow(hwnd))

    def _foreground_window_title(self) -> str:
        if os.name != "nt":
            return ""
        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return ""
        title_length = user32.GetWindowTextLengthW(hwnd)
        if title_length <= 0:
            return ""
        buffer = ctypes.create_unicode_buffer(title_length + 1)
        user32.GetWindowTextW(hwnd, buffer, title_length + 1)
        return buffer.value.strip()

    def _send_text_input(self, text: str) -> None:
        if os.name != "nt":
            raise RuntimeError("Desktop input requires Windows.")
        user32 = ctypes.windll.user32

        class KEYBDINPUT(ctypes.Structure):
            _fields_ = (
                ("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
            )

        class INPUT(ctypes.Structure):
            class _INPUTUNION(ctypes.Union):
                _fields_ = (("ki", KEYBDINPUT),)

            _anonymous_ = ("u",)
            _fields_ = (("type", ctypes.c_ulong), ("u", _INPUTUNION))

        INPUT_KEYBOARD = 1
        KEYEVENTF_KEYUP = 0x0002
        KEYEVENTF_UNICODE = 0x0004
        for character in text:
            down = INPUT(type=INPUT_KEYBOARD, ki=KEYBDINPUT(0, ord(character), KEYEVENTF_UNICODE, 0, None))
            up = INPUT(type=INPUT_KEYBOARD, ki=KEYBDINPUT(0, ord(character), KEYEVENTF_UNICODE | KEYEVENTF_KEYUP, 0, None))
            user32.SendInput(1, ctypes.byref(down), ctypes.sizeof(INPUT))
            user32.SendInput(1, ctypes.byref(up), ctypes.sizeof(INPUT))

    def _send_key_chord(self, keys: tuple[str, ...]) -> None:
        if os.name != "nt":
            raise RuntimeError("Desktop input requires Windows.")
        normalized = tuple(str(key).strip().lower() for key in keys if str(key).strip())
        if not normalized:
            raise ValueError("Desktop key input requires at least one key.")
        user32 = ctypes.windll.user32
        KEYEVENTF_KEYUP = 0x0002
        vk_codes = [self._virtual_key_code(key) for key in normalized]
        for vk_code in vk_codes:
            user32.keybd_event(vk_code, 0, 0, 0)
        for vk_code in reversed(vk_codes):
            user32.keybd_event(vk_code, 0, KEYEVENTF_KEYUP, 0)

    def _virtual_key_code(self, key_name: str) -> int:
        named_keys = {
            "alt": 0x12,
            "backspace": 0x08,
            "ctrl": 0x11,
            "delete": 0x2E,
            "down": 0x28,
            "end": 0x23,
            "enter": 0x0D,
            "esc": 0x1B,
            "escape": 0x1B,
            "home": 0x24,
            "left": 0x25,
            "pagedown": 0x22,
            "pageup": 0x21,
            "right": 0x27,
            "shift": 0x10,
            "space": 0x20,
            "tab": 0x09,
            "up": 0x26,
            "win": 0x5B,
        }
        if key_name in named_keys:
            return named_keys[key_name]
        if re.fullmatch(r"f([1-9]|1[0-2])", key_name):
            return 0x70 + int(key_name[1:]) - 1
        if len(key_name) == 1:
            if key_name.isalpha():
                return ord(key_name.upper())
            if key_name.isdigit():
                return ord(key_name)
        raise ValueError(f"Unsupported desktop key '{key_name}'.")

    def _cursor_position(self) -> tuple[int, int]:
        if os.name != "nt":
            return 0, 0

        class POINT(ctypes.Structure):
            _fields_ = (("x", ctypes.c_long), ("y", ctypes.c_long))

        point = POINT()
        ctypes.windll.user32.GetCursorPos(ctypes.byref(point))
        return int(point.x), int(point.y)

    def _set_cursor_position(self, x: int, y: int) -> None:
        if os.name != "nt":
            raise RuntimeError("Desktop input requires Windows.")
        ctypes.windll.user32.SetCursorPos(int(x), int(y))

    def _send_mouse_click(self, button: str = "left") -> None:
        if os.name != "nt":
            raise RuntimeError("Desktop input requires Windows.")
        user32 = ctypes.windll.user32
        button_name = button.strip().lower() or "left"
        mapping = {
            "left": (0x0002, 0x0004),
            "right": (0x0008, 0x0010),
            "middle": (0x0020, 0x0040),
        }
        if button_name not in mapping:
            raise ValueError(f"Unsupported mouse button '{button_name}'.")
        down_flag, up_flag = mapping[button_name]
        user32.mouse_event(down_flag, 0, 0, 0, 0)
        user32.mouse_event(up_flag, 0, 0, 0, 0)

    @staticmethod
    def _decode_preview(raw_bytes: bytes) -> str:
        return raw_bytes.decode("utf-8", errors="replace")

    def _read_text_preview(
        self,
        path: Path,
        *,
        max_bytes: int,
    ) -> tuple[str, bool, int]:
        file_size = path.stat().st_size
        with path.open("rb") as handle:
            preview_bytes = handle.read(max_bytes + 1)
        truncated = len(preview_bytes) > max_bytes
        preview = self._decode_preview(preview_bytes[:max_bytes])
        return preview, truncated, file_size

    def _read_output_preview(
        self,
        path: Path,
        *,
        max_bytes: int,
    ) -> tuple[str, bool, int]:
        file_size = path.stat().st_size if path.exists() else 0
        with path.open("rb") as handle:
            preview_bytes = handle.read(max_bytes + 1)
        truncated = len(preview_bytes) > max_bytes
        preview = self._decode_preview(preview_bytes[:max_bytes])
        return preview, truncated, file_size

    def _resolve_root(self, raw_path: str) -> Path:
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = self.workspace_root / candidate
        return candidate.resolve(strict=False)

    @staticmethod
    def _failed_result(
        request: CapabilityRequest,
        *,
        executor_kind: str,
        detail: str,
        warnings: tuple[str, ...] = (),
    ) -> CapabilityExecutionResult:
        return CapabilityExecutionResult(
            request_id=request.request_id,
            capability_type=request.capability_type,
            action_name=request.action_name(),
            status=CapabilityExecutionStatus.FAILED,
            summary=f"Capability request '{request.action_name()}' failed during execution.",
            detail=detail,
            executor_kind=executor_kind,
            output_ref=request.target_summary(),
            warnings=warnings,
            metadata={"target": request.target_summary(), "summary": request.summary},
            completed_at=utc_now(),
        )
