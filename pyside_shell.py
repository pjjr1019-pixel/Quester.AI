"""Premium PySide6 shell host for the local desktop UI migration."""

from __future__ import annotations

import math
import os
import re
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from data_structures import DashboardAppState, ShellState, UserSettingsProfile

try:  # pragma: no cover - dependency-gated path
    from PySide6 import QtCore, QtGui, QtWidgets
except Exception:  # pragma: no cover - dependency-gated path
    QtCore = None
    QtGui = None
    QtWidgets = None


class PySideShellUnavailableError(RuntimeError):
    """Raised when the optional PySide6 shell is requested without the desktop extra."""


def pyside6_available() -> bool:
    """Return whether the optional PySide6 desktop shell dependency is installed."""
    return QtWidgets is not None


if QtWidgets is not None:  # pragma: no branch

    @dataclass(frozen=True)
    class ShellThemeTokens:
        accent: str
        bg: str
        highlight: str
        panel: str
        panel_alt: str
        glass: str
        edge: str
        text: str
        muted: str
        warning: str
        danger: str
        success: str
        background_top: str
        background_mid: str
        background_bottom: str
        haze: str
        reflection: str
        chip_bg: str
        chip_bg_active: str
        notification_bg: str
        notification_border: str
        accent_rail: str
        timeline_info: str
        timeline_warning: str
        timeline_error: str
        button_bg: str
        button_hover: str
        button_focus: str
        card_active: str
        card_coding: str
        card_final: str
        card_warning: str
        font_ui: str
        font_status: str
        font_mono: str

        def __getitem__(self, key: str) -> str:
            return str(getattr(self, key))

    _MODE_PALETTE_MAP = {
        "offline": "slate_blue",
        "error": "warning_red",
        "speaking": "vivid_blue",
        "listening": "vivid_blue",
        "responding": "cyan_white",
        "planner": "cyan_blue",
        "researcher_web": "amber_gold",
        "researcher_local": "blue_gold",
        "critic": "violet_magenta",
        "verification": "cyan_white",
        "compressor": "white_gold",
        "reasoner_fast": "focused_yellow",
        "reasoner_deep": "deep_red",
        "code_planning": "cyan_blue",
        "code_generating": "focused_yellow",
        "code_refactoring": "cyan_white",
        "code_testing": "green_blue",
        "code_debugging": "warning_red",
        "code_reviewing": "violet_magenta",
        "code_indexing": "blue_gold",
        "code_practicing": "indigo_gold",
        "code_learning": "white_gold",
        "code_regression": "warning_red",
        "idle": "calm_blue",
    }
    _CODING_STATE_PALETTE_MAP = {
        "planning": "cyan_blue",
        "generating": "focused_yellow",
        "testing": "green_blue",
        "debugging": "deep_red",
        "reviewing": "violet_magenta",
        "refactoring": "cyan_white",
        "indexing": "blue_gold",
        "practicing": "indigo_gold",
        "learning": "white_gold",
        "regression_detected": "warning_red",
        "validated": "green_blue",
    }
    _PANEL_KICKER_MAP = {
        "timeline": "TIMELINE",
        "evidence": "EVIDENCE",
        "provenance": "SOURCE MAP",
        "compressor": "COMPRESSION",
        "optimizer": "OPTIMIZER",
        "control": "CONTROL",
        "runtime": "RUNTIME",
        "practice": "PRACTICE",
        "pattern": "PATTERNS",
        "memory": "MEMORY",
        "validation": "VALIDATION",
        "metrics": "METRICS",
        "session": "SESSION",
    }

    def _normalized_shell_value(value: Any) -> str:
        return str(value or "").strip().lower()

    def _is_live_shell_state(value: Any) -> bool:
        return _normalized_shell_value(value) not in {
            "",
            "idle",
            "inactive",
            "disabled",
            "offline",
            "ready",
            "complete",
            "completed",
            "succeeded",
            "success",
            "passed",
        }

    def _panel_kicker(title: str) -> str:
        normalized = _normalized_shell_value(title)
        for key, label in _PANEL_KICKER_MAP.items():
            if key in normalized:
                return label
        if not normalized:
            return "PANEL"
        return normalized.split()[0].upper()

    def _effective_palette_key(shell_state: ShellState) -> str:
        explicit = _normalized_shell_value(shell_state.orb_palette)
        orb_mode = _normalized_shell_value(shell_state.orb_mode)
        coding_state = _normalized_shell_value(shell_state.coding_state)
        verifier_state = _normalized_shell_value(shell_state.verifier_state)
        retrieval_state = _normalized_shell_value(shell_state.retrieval_state)
        compression_state = _normalized_shell_value(shell_state.compression_state)
        optimizer_state = _normalized_shell_value(shell_state.optimizer_state)
        quality_gate_state = _normalized_shell_value(shell_state.quality_gate_state)
        long_horizon_state = _normalized_shell_value(shell_state.long_horizon_state)
        capability_state = _normalized_shell_value(shell_state.capability_session_state)
        cloud_helper_state = _normalized_shell_value(shell_state.cloud_helper_state)
        speaking_state = _normalized_shell_value(shell_state.speaking_state)
        active_tools = {_normalized_shell_value(tool) for tool in shell_state.active_tools}
        is_generic = explicit in {"", "calm_blue", "slate_blue"}

        if (
            shell_state.degraded_reason
            or shell_state.fallback_reason
            or orb_mode in {"error", "code_debugging", "code_regression"}
        ):
            return "warning_red"
        if quality_gate_state in {"failed", "blocked", "rejected", "regression_detected"}:
            return "deep_red"
        if shell_state.approval_pending:
            return "amber_gold"
        if speaking_state in {"active", "speaking"} or orb_mode in {"speaking", "listening"}:
            return "vivid_blue"
        if shell_state.workspace_mode == "coding_workspace":
            if coding_state in _CODING_STATE_PALETTE_MAP:
                return _CODING_STATE_PALETTE_MAP[coding_state]
            if quality_gate_state in {"verifying", "verified", "passed"}:
                return "green_blue"
            if not is_generic:
                return explicit
            return "focused_yellow"
        if not is_generic:
            return explicit
        if orb_mode in _MODE_PALETTE_MAP:
            return _MODE_PALETTE_MAP[orb_mode]
        if verifier_state in {"running", "verifying", "verified"} or quality_gate_state in {
            "verifying",
            "verified",
            "reviewing",
        }:
            return "cyan_white"
        if _is_live_shell_state(optimizer_state) or _is_live_shell_state(compression_state):
            return "white_gold"
        if _is_live_shell_state(retrieval_state):
            return "amber_gold" if "web" in active_tools else "blue_gold"
        if _is_live_shell_state(long_horizon_state):
            return "blue_gold"
        if _is_live_shell_state(capability_state):
            return "violet_magenta"
        if _is_live_shell_state(cloud_helper_state):
            return "indigo_gold"
        if "web" in active_tools:
            return "amber_gold"
        if active_tools:
            return "calm_blue"
        return explicit or "calm_blue"

    def _theme(shell_state: ShellState, ui: dict[str, Any] | None = None) -> ShellThemeTokens:
        palette_map = {
            "calm_blue": ("#61b7ff", "#07111f", "#d9f5ff"),
            "vivid_blue": ("#59c7ff", "#08121f", "#ecfbff"),
            "cyan_blue": ("#79d1ff", "#071722", "#f0fdff"),
            "blue_gold": ("#8ab8ff", "#101522", "#fff4db"),
            "green_blue": ("#73d7c4", "#08171a", "#e7fff8"),
            "amber_gold": ("#ffb85a", "#1a120b", "#fff4cb"),
            "focused_yellow": ("#ffd760", "#171208", "#fff8d8"),
            "deep_red": ("#ff646e", "#19080d", "#ffe0d5"),
            "indigo_gold": ("#9d9cff", "#0f0d1b", "#fff0c9"),
            "violet_magenta": ("#c98bff", "#12091a", "#ffe8ff"),
            "white_gold": ("#ffe4a1", "#17140c", "#ffffff"),
            "cyan_white": ("#b5ecff", "#09131c", "#ffffff"),
            "warning_red": ("#ff6d72", "#1b090c", "#ffdede"),
            "slate_blue": ("#93a9d3", "#0a1019", "#e7eefc"),
        }
        accent, bg, highlight = palette_map.get(_effective_palette_key(shell_state), palette_map["calm_blue"])
        is_coding = shell_state.workspace_mode == "coding_workspace"
        is_warning = bool(shell_state.degraded_reason or shell_state.fallback_reason)
        font_ui = "'Segoe UI Variable Text', 'Segoe UI', 'Trebuchet MS', sans-serif"
        font_status = "'Bahnschrift SemiBold', 'Segoe UI Semibold', 'Segoe UI', sans-serif"
        font_mono = "'Cascadia Code', 'Consolas', monospace"
        background_top = "#081525" if not is_coding else "#211106"
        background_mid = bg
        background_bottom = "#030811" if not is_coding else "#120905"
        panel = "rgba(8, 18, 34, 0.76)" if not is_coding else "rgba(28, 16, 7, 0.80)"
        panel_alt = "rgba(14, 28, 52, 0.72)" if not is_coding else "rgba(48, 28, 12, 0.76)"
        card_warning = "rgba(46, 21, 20, 0.94)" if is_warning else panel_alt
        return ShellThemeTokens(
            accent=accent,
            bg=bg,
            highlight=highlight,
            panel=panel,
            panel_alt=panel_alt,
            glass="rgba(255, 255, 255, 0.055)",
            edge="rgba(154, 205, 255, 0.16)" if not is_coding else "rgba(255, 193, 116, 0.16)",
            text="#eef5ff",
            muted="#b6c2d7",
            warning="#f6c36b",
            danger="#ff7e8f",
            success="#8fe0b3",
            background_top=background_top,
            background_mid=background_mid,
            background_bottom=background_bottom,
            haze=f"rgba({QtGui.QColor(accent).red()}, {QtGui.QColor(accent).green()}, {QtGui.QColor(accent).blue()}, 0.16)",
            reflection="rgba(255, 255, 255, 0.08)",
            chip_bg="rgba(255, 255, 255, 0.05)",
            chip_bg_active="rgba(255, 255, 255, 0.12)",
            notification_bg="rgba(17, 26, 42, 0.86)" if not is_coding else "rgba(38, 22, 11, 0.88)",
            notification_border="rgba(255, 255, 255, 0.16)",
            accent_rail=accent,
            timeline_info="#d9e7ff",
            timeline_warning="#f6c36b",
            timeline_error="#ff98a5",
            button_bg="rgba(255, 255, 255, 0.05)",
            button_hover="rgba(255, 255, 255, 0.14)",
            button_focus="rgba(255, 255, 255, 0.20)",
            card_active="rgba(10, 26, 46, 0.84)" if not is_warning else card_warning,
            card_coding="rgba(39, 24, 11, 0.88)",
            card_final="rgba(13, 28, 43, 0.88)",
            card_warning=card_warning,
            font_ui=font_ui,
            font_status=font_status,
            font_mono=font_mono,
        )

    _THEME_COLOR_FIELDS = (
        "accent",
        "bg",
        "highlight",
        "panel",
        "panel_alt",
        "glass",
        "edge",
        "text",
        "muted",
        "warning",
        "danger",
        "success",
        "background_top",
        "background_mid",
        "background_bottom",
        "haze",
        "reflection",
        "chip_bg",
        "chip_bg_active",
        "notification_bg",
        "notification_border",
        "accent_rail",
        "timeline_info",
        "timeline_warning",
        "timeline_error",
        "button_bg",
        "button_hover",
        "button_focus",
        "card_active",
        "card_coding",
        "card_final",
        "card_warning",
    )
    _THEME_FONT_FIELDS = ("font_ui", "font_status", "font_mono")
    _RGBA_COLOR_RE = re.compile(
        r"rgba?\(\s*(?P<r>\d{1,3})\s*,\s*(?P<g>\d{1,3})\s*,\s*(?P<b>\d{1,3})(?:\s*,\s*(?P<a>[0-9.]+))?\s*\)",
        re.IGNORECASE,
    )
    _ITEM_DETAIL_ROLE = int(QtCore.Qt.ItemDataRole.UserRole) + 1

    def _parse_css_color(value: str) -> QtGui.QColor:
        normalized = str(value or "").strip()
        match = _RGBA_COLOR_RE.fullmatch(normalized)
        if match is not None:
            alpha_text = match.group("a")
            alpha = 255
            if alpha_text is not None:
                alpha_value = float(alpha_text)
                alpha = round(alpha_value * 255.0) if alpha_value <= 1.0 else round(alpha_value)
            return QtGui.QColor(
                max(0, min(255, int(match.group("r")))),
                max(0, min(255, int(match.group("g")))),
                max(0, min(255, int(match.group("b")))),
                max(0, min(255, int(alpha))),
            )
        color = QtGui.QColor(normalized)
        return color if color.isValid() else QtGui.QColor("#000000")

    def _serialize_css_color(color: QtGui.QColor) -> str:
        format_name = (
            QtGui.QColor.NameFormat.HexArgb
            if color.alpha() < 255
            else QtGui.QColor.NameFormat.HexRgb
        )
        return color.name(format_name)

    def _blend_qcolor(current: QtGui.QColor, target: QtGui.QColor, factor: float) -> QtGui.QColor:
        if factor >= 1.0:
            return QtGui.QColor(target)
        channel_deltas = (
            abs(current.red() - target.red()),
            abs(current.green() - target.green()),
            abs(current.blue() - target.blue()),
            abs(current.alpha() - target.alpha()),
        )
        if max(channel_deltas) <= 2:
            return QtGui.QColor(target)

        def _next_channel(current_value: int, target_value: int) -> int:
            if abs(current_value - target_value) <= 2:
                return target_value
            delta = target_value - current_value
            step = round(delta * factor)
            if step == 0:
                step = 1 if delta > 0 else -1
            return current_value + step

        return QtGui.QColor(
            _next_channel(current.red(), target.red()),
            _next_channel(current.green(), target.green()),
            _next_channel(current.blue(), target.blue()),
            _next_channel(current.alpha(), target.alpha()),
        )

    def _blend_theme(current: ShellThemeTokens, target: ShellThemeTokens, factor: float) -> ShellThemeTokens:
        blended = {
            field_name: _serialize_css_color(
                _blend_qcolor(
                    _parse_css_color(getattr(current, field_name)),
                    _parse_css_color(getattr(target, field_name)),
                    factor,
                )
            )
            for field_name in _THEME_COLOR_FIELDS
        }
        for field_name in _THEME_FONT_FIELDS:
            blended[field_name] = getattr(target, field_name)
        return ShellThemeTokens(**blended)

    def _clear(layout: QtWidgets.QLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            child_layout = item.layout()
            if widget is not None:
                widget.deleteLater()
            elif child_layout is not None:
                _clear(child_layout)

    def _format_timestamp(value: Any) -> str:
        if isinstance(value, datetime):
            return value.strftime("%Y-%m-%d %H:%M")
        return "(unknown)"

    def _mean(values: list[float]) -> float:
        return sum(values) / max(1, len(values))

    _LEFT_DOCK_WIDE_THRESHOLD = 1240
    _RIGHT_DOCK_WIDE_THRESHOLD = 1360
    _BOTTOM_DOCK_WIDE_THRESHOLD = 1460

    class ShellAnimationClock(QtCore.QObject):
        """One shared animation clock for orb, chips, and notification pulses."""

        tick = QtCore.Signal(float)

        def __init__(self, parent: QtCore.QObject | None = None) -> None:
            super().__init__(parent)
            self._phase = 0.0
            self._timer = QtCore.QTimer(self)
            self._timer.timeout.connect(self._on_tick)
            self._timer.start(33)

        @property
        def phase(self) -> float:
            return self._phase

        def timer(self) -> QtCore.QTimer:
            return self._timer

        def apply_ui_preferences(self, ui: dict[str, Any]) -> None:
            frame_cap = int(ui.get("animation_frame_cap", 30) or 30)
            interval_ms = max(8, int(1000 / max(10, min(120, frame_cap))))
            if bool(ui.get("reduced_motion", False)):
                interval_ms = max(interval_ms, 66)
            elif bool(ui.get("reduced_effects_mode", False)) or bool(ui.get("low_resource_mode", False)):
                interval_ms = max(interval_ms, 50)
            self._timer.start(interval_ms)

        @QtCore.Slot()
        def _on_tick(self) -> None:
            self._phase = (self._phase + 0.045) % (math.tau * 8.0)
            self.tick.emit(self._phase)

        def stop(self) -> None:
            if self._timer.isActive():
                self._timer.stop()

    class OrbWidget(QtWidgets.QWidget):
        """Custom-painted orb driven by the shell state."""

        def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
            super().__init__(parent)
            self._shell_state = ShellState()
            self._minimal = False
            self._reduced_motion = False
            self._reduced_effects = False
            self._simple_orb = False
            self._ambient_reactivity = True
            self._orb_scale = 1.0
            self._animation_intensity = 1.0
            self._display_intensity = 0.12
            self._particle_density = "balanced"
            self._phase = 0.0
            self._clock: ShellAnimationClock | None = None
            initial_theme = _theme(self._shell_state)
            self._display_primary = _parse_css_color(initial_theme["accent"])
            self._display_shadow = _parse_css_color(initial_theme["bg"])
            self._display_highlight = _parse_css_color(initial_theme["highlight"])
            self._target_primary = QtGui.QColor(self._display_primary)
            self._target_shadow = QtGui.QColor(self._display_shadow)
            self._target_highlight = QtGui.QColor(self._display_highlight)
            self.setMinimumHeight(360)
            self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        def set_shell_state(self, shell_state: ShellState) -> None:
            self._shell_state = shell_state
            self._sync_palette_targets()
            self.update()

        def set_ui_preferences(self, ui: dict[str, Any]) -> None:
            self._minimal = (
                bool(ui.get("lightweight_mode", False))
                or bool(ui.get("low_resource_mode", False))
                or str(
                ui.get("shell_preset", "balanced")
            ).lower() == "minimal"
            )
            self._reduced_motion = bool(ui.get("reduced_motion", False))
            self._reduced_effects = bool(ui.get("reduced_effects_mode", False))
            self._simple_orb = bool(ui.get("simple_orb_mode", False))
            self._ambient_reactivity = bool(ui.get("ambient_reactivity", True))
            self._orb_scale = max(0.6, min(1.6, float(ui.get("orb_size", 100) or 100) / 100.0))
            self._animation_intensity = max(
                0.2, min(1.5, float(ui.get("animation_intensity", 1.0) or 1.0))
            )
            self._particle_density = str(ui.get("particle_density", "balanced") or "balanced").lower()
            if self._reduced_motion:
                self._display_primary = QtGui.QColor(self._target_primary)
                self._display_shadow = QtGui.QColor(self._target_shadow)
                self._display_highlight = QtGui.QColor(self._target_highlight)
            self.update()

        def _sync_palette_targets(self) -> None:
            theme = _theme(self._shell_state)
            self._target_primary = _parse_css_color(theme["accent"])
            self._target_shadow = _parse_css_color(theme["bg"])
            self._target_highlight = _parse_css_color(theme["highlight"])
            if self._reduced_motion:
                self._display_primary = QtGui.QColor(self._target_primary)
                self._display_shadow = QtGui.QColor(self._target_shadow)
                self._display_highlight = QtGui.QColor(self._target_highlight)

        def attach_animation_clock(self, clock: ShellAnimationClock) -> None:
            if self._clock is clock:
                return
            if self._clock is not None:
                try:
                    self._clock.tick.disconnect(self._on_animation_tick)
                except (RuntimeError, TypeError):
                    pass
            self._clock = clock
            clock.tick.connect(self._on_animation_tick)
            self._phase = clock.phase

        @QtCore.Slot(float)
        def _on_animation_tick(self, phase: float) -> None:
            self._phase = phase
            target_intensity = max(0.1, min(1.0, self._shell_state.orb_intensity * self._animation_intensity))
            self._display_intensity += (target_intensity - self._display_intensity) * 0.18
            palette_factor = 1.0 if self._reduced_motion else 0.28 if self._minimal or self._reduced_effects else 0.16
            self._display_primary = _blend_qcolor(self._display_primary, self._target_primary, palette_factor)
            self._display_shadow = _blend_qcolor(self._display_shadow, self._target_shadow, palette_factor)
            self._display_highlight = _blend_qcolor(self._display_highlight, self._target_highlight, palette_factor)
            self.update()

        def shutdown(self) -> None:
            if self._clock is not None:
                try:
                    self._clock.tick.disconnect(self._on_animation_tick)
                except (RuntimeError, TypeError):
                    pass
                self._clock = None

        def _palette(self) -> tuple[QtGui.QColor, QtGui.QColor, QtGui.QColor]:
            return (
                QtGui.QColor(self._display_primary),
                QtGui.QColor(self._display_shadow),
                QtGui.QColor(self._display_highlight),
            )

        def _effect_profile(self) -> dict[str, bool]:
            effects = self._shell_state.orb_effects
            approval = bool(self._shell_state.approval_pending or effects.approval_hold)
            degraded = bool(self._shell_state.degraded_reason or effects.degraded_undertone)
            speaking = bool(
                self._shell_state.orb_mode == "speaking"
                or self._shell_state.speaking_state in {"active", "speaking"}
            )
            verification = bool(effects.verification_lock_pending and not approval)
            checkpoint = bool(effects.checkpoint_pulse_pending and not approval and not speaking)
            insight = bool(
                (effects.insight_flash_pending or "insight_flash" in effects.transient_effects)
                and not approval
                and not verification
            )
            consensus = bool(
                (effects.consensus_shimmer_pending or "consensus_shimmer" in effects.transient_effects)
                and not approval
                and not verification
            )
            optimizer = bool(
                self._shell_state.optimizer_state not in {"", "idle", "inactive"}
                and not approval
                and not degraded
            )
            return {
                "approval": approval,
                "degraded": degraded,
                "speaking": speaking and not approval,
                "verification": verification,
                "checkpoint": checkpoint,
                "insight": insight,
                "consensus": consensus,
                "optimizer": optimizer,
                "show_telemetry": not approval,
            }

        def _particle_budget(self, mode: str | None = None) -> int:
            density_limits = {"minimal": 4, "balanced": 8, "immersive": 12}
            base_budget = density_limits.get(self._particle_density, density_limits["balanced"])
            if self._minimal or self._reduced_effects:
                base_budget = max(3, base_budget - 2)
            mode_multipliers = {
                "sparse": 1,
                "inward": 2,
                "dense_orbit": 3,
                "spark": 2,
                "halo": 2,
            }
            particle_mode = str(mode or self._shell_state.particle_mode or "sparse").lower()
            return max(0, min(36, base_budget * mode_multipliers.get(particle_mode, 1)))

        def _role_constellation_points(
            self,
            center: QtCore.QPointF,
            radius: float,
        ) -> list[tuple[QtCore.QPointF, str]]:
            roles = [
                *tuple(self._shell_state.active_roles),
                *tuple(self._shell_state.active_model_roles),
            ]
            deduped_roles = [role for role in dict.fromkeys(str(role).strip() for role in roles) if role]
            if not deduped_roles:
                return []
            points: list[tuple[QtCore.QPointF, str]] = []
            orbit_radius = radius * 1.78
            for index, role_name in enumerate(deduped_roles[:8]):
                angle = (math.tau * index / max(1, min(8, len(deduped_roles)))) + (self._phase * 0.22)
                wobble = 0.92 if index % 2 == 0 else 1.04
                points.append(
                    (
                        QtCore.QPointF(
                            center.x() + (math.cos(angle) * orbit_radius * wobble),
                            center.y() + (math.sin(angle) * orbit_radius * wobble * 0.72),
                        ),
                        role_name,
                    )
                )
            return points

        def _particle_specs(
            self,
            center: QtCore.QPointF,
            radius: float,
        ) -> list[tuple[QtCore.QPointF, float, float]]:
            particle_mode = str(self._shell_state.particle_mode or "sparse").lower()
            particle_count = self._particle_budget(particle_mode)
            if particle_count <= 0:
                return []

            specs: list[tuple[QtCore.QPointF, float, float]] = []
            for index in range(particle_count):
                progress = index / max(1, particle_count)
                phase = self._phase + (index * 0.41)
                if particle_mode == "inward":
                    angle = (math.tau * progress * 1.5) + (self._phase * 0.35)
                    path_progress = (phase * 0.16) % 1.0
                    orbit_radius = radius * (2.2 - (1.25 * path_progress))
                    point = QtCore.QPointF(
                        center.x() + (math.cos(angle) * orbit_radius),
                        center.y() + (math.sin(angle) * orbit_radius * 0.72),
                    )
                    alpha = 0.26 + (0.26 * (1.0 - path_progress))
                    size = 1.4 + (1.2 * (1.0 - path_progress))
                elif particle_mode == "dense_orbit":
                    band = 1.55 if index % 2 == 0 else 1.95
                    angle = (math.tau * progress * 2.0) + (self._phase * (0.48 if index % 2 == 0 else -0.36))
                    point = QtCore.QPointF(
                        center.x() + (math.cos(angle) * radius * band),
                        center.y() + (math.sin(angle) * radius * band * 0.76),
                    )
                    alpha = 0.20 + (0.18 * (0.5 + (math.sin(phase) * 0.5)))
                    size = 1.6 + (1.0 * (index % 3 == 0))
                elif particle_mode == "spark":
                    angle = (math.tau * progress) + (self._phase * 0.85)
                    burst = 1.72 + (0.14 * math.sin(phase * 1.6))
                    point = QtCore.QPointF(
                        center.x() + (math.cos(angle) * radius * burst),
                        center.y() + (math.sin(angle) * radius * burst * 0.78),
                    )
                    alpha = 0.30 + (0.30 * (0.5 + (math.sin(phase * 1.4) * 0.5)))
                    size = 1.8 if index % 4 else 2.6
                elif particle_mode == "halo":
                    angle = (math.tau * progress) + (self._phase * 1.05)
                    halo_radius = radius * (1.98 + (0.08 * math.sin(phase)))
                    point = QtCore.QPointF(
                        center.x() + (math.cos(angle) * halo_radius),
                        center.y() + (math.sin(angle) * halo_radius * 0.82),
                    )
                    alpha = 0.24 + (0.26 * (0.5 + (math.sin(phase * 1.8) * 0.5)))
                    size = 1.8 + (0.8 * (index % 5 == 0))
                else:
                    angle = (math.tau * progress) + (self._phase * 0.24)
                    orbit_radius = radius * (1.74 + (0.08 * math.sin(phase)))
                    point = QtCore.QPointF(
                        center.x() + (math.cos(angle) * orbit_radius),
                        center.y() + (math.sin(angle) * orbit_radius * 0.76),
                    )
                    alpha = 0.16 + (0.12 * (0.5 + (math.sin(phase) * 0.5)))
                    size = 1.4
                specs.append((point, alpha, size))
            return specs

        def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # pragma: no cover - dependency-gated path
            started_at = time.perf_counter()
            del event
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.Antialiasing)
            painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)

            rect = self.rect()
            theme = _theme(self._shell_state)
            painter.fillRect(rect, QtCore.Qt.GlobalColor.transparent)

            primary, shadow, highlight = self._palette()
            effect_profile = self._effect_profile()
            intensity = max(0.1, min(1.0, self._display_intensity))
            breathe = 1.0 + (math.sin(self._phase) * 0.04 * intensity)
            float_offset = 0.0 if self._reduced_motion else math.sin(self._phase * 0.72) * 8.0 * intensity
            center = QtCore.QPointF(rect.center().x(), rect.center().y() - (rect.height() * 0.08) + float_offset)
            radius = min(rect.width(), rect.height()) * 0.21 * breathe * self._orb_scale
            pulse = 1.0 if self._reduced_motion else 1.0 + (math.sin(self._phase * 1.9) * 0.03 * intensity)

            if effect_profile["degraded"]:
                degraded_glow = QtGui.QRadialGradient(center, radius * 2.3)
                degraded_glow.setColorAt(0.0, QtGui.QColor(255, 109, 114, 52 if not self._reduced_effects else 30))
                degraded_glow.setColorAt(0.7, QtGui.QColor(140, 22, 33, 26 if not self._reduced_effects else 14))
                degraded_glow.setColorAt(1.0, QtGui.QColor(0, 0, 0, 0))
                painter.setPen(QtCore.Qt.PenStyle.NoPen)
                painter.setBrush(degraded_glow)
                painter.drawEllipse(center, radius * 2.15, radius * 1.85)

            if self._ambient_reactivity and not self._reduced_effects:
                ambient = QtGui.QRadialGradient(center, radius * 1.9)
                ambient.setColorAt(0.0, QtGui.QColor(primary.red(), primary.green(), primary.blue(), int(72 * intensity)))
                ambient.setColorAt(0.65, QtGui.QColor(primary.red(), primary.green(), primary.blue(), int(24 * intensity)))
                ambient.setColorAt(1.0, QtGui.QColor(0, 0, 0, 0))
                painter.setPen(QtCore.Qt.PenStyle.NoPen)
                painter.setBrush(ambient)
                painter.drawEllipse(center, radius * 1.95, radius * 1.65)

            particle_specs = self._particle_specs(center, radius)
            if effect_profile["show_telemetry"] and particle_specs:
                particle_color = QtGui.QColor(primary.red(), primary.green(), primary.blue(), 64)
                painter.setPen(QtCore.Qt.PenStyle.NoPen)
                for point, alpha, size in particle_specs:
                    particle_color.setAlpha(max(18, min(180, int(255 * alpha * intensity))))
                    painter.setBrush(particle_color)
                    painter.drawEllipse(point, size, size)

            halo_pen = QtGui.QPen(QtGui.QColor(primary.red(), primary.green(), primary.blue(), int(160 * intensity)))
            halo_pen.setWidthF(max(2.0, radius * 0.03))
            painter.setPen(halo_pen)
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.drawEllipse(center, radius * 1.12, radius * 1.12)

            orb = QtGui.QRadialGradient(center.x() - (radius * 0.18), center.y() - (radius * 0.26), radius * 1.35)
            orb.setColorAt(0.0, highlight)
            orb.setColorAt(0.28, primary.lighter(135))
            orb.setColorAt(0.72, primary.darker(150))
            orb.setColorAt(1.0, shadow)
            painter.setBrush(orb)
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.drawEllipse(center, radius, radius)

            inner_shimmer = QtGui.QRadialGradient(
                center.x() - (radius * 0.22),
                center.y() - (radius * 0.24),
                radius * 0.92,
            )
            inner_shimmer.setColorAt(0.0, QtGui.QColor(255, 255, 255, int(96 * intensity)))
            inner_shimmer.setColorAt(0.34, QtGui.QColor(highlight.red(), highlight.green(), highlight.blue(), int(54 * intensity)))
            inner_shimmer.setColorAt(1.0, QtGui.QColor(255, 255, 255, 0))
            painter.setBrush(inner_shimmer)
            painter.drawEllipse(center, radius * 0.86, radius * 0.86)

            reflection = QtGui.QLinearGradient(
                center.x() - radius * 0.45,
                center.y() - radius * 0.85,
                center.x() + radius * 0.25,
                center.y() - radius * 0.15,
            )
            reflection.setColorAt(0.0, QtGui.QColor(255, 255, 255, int(155 * intensity)))
            reflection.setColorAt(1.0, QtGui.QColor(255, 255, 255, 0))
            painter.setBrush(reflection)
            painter.drawEllipse(
                QtCore.QRectF(
                    center.x() - radius * 0.58,
                    center.y() - radius * 0.92,
                    radius * 0.86,
                    radius * 0.7,
                )
            )

            shell_surface_pen = QtGui.QPen(QtGui.QColor(255, 255, 255, int(58 * intensity)))
            shell_surface_pen.setWidthF(max(1.0, radius * 0.012))
            painter.setPen(shell_surface_pen)
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.drawEllipse(center, radius * 0.94, radius * 0.94)

            ring_pen = QtGui.QPen(QtGui.QColor(highlight.red(), highlight.green(), highlight.blue(), int(150 * intensity)))
            ring_pen.setWidthF(max(1.0, radius * 0.015))
            painter.setPen(ring_pen)
            sweep_angle = 220 if self._shell_state.ring_mode in {"scan", "focus"} else 320
            painter.drawArc(
                QtCore.QRectF(
                    center.x() - radius * 1.34,
                    center.y() - radius * 1.34,
                    radius * 2.68,
                    radius * 2.68,
                ),
                int((self._phase * 30.0) % 360 * 16),
                int(sweep_angle * 16),
            )

            if not self._simple_orb:
                orbit_specs = (
                    (18.0 + (math.sin(self._phase * 0.48) * 3.0), QtGui.QColor(primary.red(), primary.green(), primary.blue(), int(180 * intensity)), radius * 1.72, radius * 0.48),
                    (-12.0 + (math.sin(self._phase * 0.62) * 4.0), QtGui.QColor(theme["warning"]), radius * 1.94, radius * 0.54),
                )
                for rotation, color, orbit_width, orbit_height in orbit_specs:
                    painter.save()
                    painter.translate(center)
                    painter.rotate(rotation)
                    orbit_pen = QtGui.QPen(color)
                    orbit_pen.setWidthF(1.3 if self._reduced_effects else 2.2)
                    painter.setPen(orbit_pen)
                    painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
                    painter.drawEllipse(
                        QtCore.QRectF(
                            -orbit_width,
                            -orbit_height,
                            orbit_width * 2.0,
                            orbit_height * 2.0,
                        )
                    )
                    painter.restore()

            core_glow = QtGui.QRadialGradient(center, radius * 0.26)
            core_glow.setColorAt(0.0, QtGui.QColor(255, 244, 214, int(255 * intensity)))
            core_glow.setColorAt(0.48, QtGui.QColor(highlight.red(), highlight.green(), highlight.blue(), int(190 * intensity)))
            core_glow.setColorAt(1.0, QtGui.QColor(255, 255, 255, 0))
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(core_glow)
            painter.drawEllipse(center, radius * 0.22, radius * 0.22)

            if effect_profile["show_telemetry"] and not self._minimal and not self._simple_orb:
                segment_limit = 3 if self._particle_density == "minimal" else 8 if self._particle_density == "immersive" else 5
                tool_segments = max(1, min(segment_limit, len(self._shell_state.active_tools)))
                segment_pen = QtGui.QPen(QtGui.QColor(theme["highlight"]))
                segment_pen.setWidthF(max(1.4, radius * 0.016))
                painter.setPen(segment_pen)
                segment_rect = QtCore.QRectF(
                    center.x() - radius * 1.36,
                    center.y() - radius * 1.36,
                    radius * 2.72,
                    radius * 2.72,
                )
                start_angle = int((self._phase * 18.0) % 360 * 16)
                for index in range(tool_segments):
                    painter.drawArc(segment_rect, start_angle + (index * 26 * 16), int(18 * 16))

                confidence_pen = QtGui.QPen(QtGui.QColor(theme["success"]))
                confidence_pen.setWidthF(max(2.0, radius * 0.018))
                painter.setPen(confidence_pen)
                confidence_ratio = (
                    0.86
                    if self._shell_state.confidence_band == "high"
                    else 0.62
                    if self._shell_state.confidence_band == "medium"
                    else 0.34
                )
                painter.drawArc(
                    QtCore.QRectF(
                        center.x() - radius * 1.52,
                        center.y() - radius * 1.52,
                        radius * 3.04,
                        radius * 3.04,
                    ),
                    120 * 16,
                    int(300 * confidence_ratio * 16),
                )

                constellation_points = self._role_constellation_points(center, radius)
                if constellation_points:
                    link_pen = QtGui.QPen(QtGui.QColor(theme["edge"]))
                    link_pen.setWidthF(1.0)
                    painter.setPen(link_pen)
                    anchor_radius = radius * 1.22
                    for point, _role_name in constellation_points:
                        direction_x = point.x() - center.x()
                        direction_y = point.y() - center.y()
                        direction_length = max(1.0, math.hypot(direction_x, direction_y))
                        anchor = QtCore.QPointF(
                            center.x() + ((direction_x / direction_length) * anchor_radius),
                            center.y() + ((direction_y / direction_length) * anchor_radius),
                        )
                        painter.drawLine(anchor, point)
                    painter.setPen(QtCore.Qt.PenStyle.NoPen)
                    for point, role_name in constellation_points:
                        marker_color = QtGui.QColor(theme["warning"] if "critic" in role_name else theme["highlight"])
                        marker_color.setAlpha(200)
                        painter.setBrush(marker_color)
                        painter.drawEllipse(point, 3.0 if self._reduced_effects else 4.2, 3.0 if self._reduced_effects else 4.2)

            if effect_profile["optimizer"]:
                optimizer_pen = QtGui.QPen(QtGui.QColor(theme["highlight"]))
                optimizer_pen.setWidthF(1.4 if self._reduced_effects else 1.8)
                optimizer_pen.setStyle(QtCore.Qt.PenStyle.DashLine)
                painter.setPen(optimizer_pen)
                painter.drawEllipse(center, radius * 1.73, radius * 1.73)

            if effect_profile["approval"]:
                hold_pen = QtGui.QPen(QtGui.QColor(theme["warning"]))
                hold_pen.setWidthF(max(2.0, radius * 0.02))
                painter.setPen(hold_pen)
                painter.drawEllipse(center, radius * 1.28, radius * 1.28)

            if effect_profile["verification"]:
                lock_pen = QtGui.QPen(QtGui.QColor(theme["success"]))
                lock_pen.setWidthF(max(1.0, radius * 0.013))
                painter.setPen(lock_pen)
                painter.drawEllipse(center, radius * 1.45, radius * 1.45)
            if effect_profile["verification"] or self._shell_state.orb_mode == "critic":
                verification_pen = QtGui.QPen(QtGui.QColor(theme["success"]))
                verification_pen.setWidthF(1.2 if self._reduced_effects else 2.0)
                painter.setPen(verification_pen)
                painter.drawArc(
                    QtCore.QRectF(
                        center.x() - radius * 1.62,
                        center.y() - radius * 1.62,
                        radius * 3.24,
                        radius * 3.24,
                    ),
                    int((self._phase * 90.0) % 360 * 16),
                    int((26 if self._reduced_effects else 42) * 16),
                )

            if self._shell_state.orb_mode == "compressor" or self._shell_state.compression_state == "active":
                compressor_pen = QtGui.QPen(QtGui.QColor(theme["warning"]))
                compressor_pen.setWidthF(1.4 if self._reduced_effects else 1.9)
                painter.setPen(compressor_pen)
                contraction = 0.86 + ((0.04 if self._reduced_effects else 0.08) * math.sin(self._phase * 2.8))
                painter.drawEllipse(center, radius * contraction, radius * contraction * pulse)

            if effect_profile["checkpoint"]:
                checkpoint_pen = QtGui.QPen(QtGui.QColor(theme["warning"]))
                checkpoint_pen.setWidthF(max(1.8, radius * 0.018))
                painter.setPen(checkpoint_pen)
                painter.drawEllipse(center, radius * 1.62, radius * 1.62)

            if effect_profile["insight"]:
                insight_pen = QtGui.QPen(QtGui.QColor(theme["highlight"]))
                insight_pen.setWidthF(1.2 if self._reduced_effects else 1.8)
                painter.setPen(insight_pen)
                painter.drawArc(
                    QtCore.QRectF(
                        center.x() - radius * 1.86,
                        center.y() - radius * 1.86,
                        radius * 3.72,
                        radius * 3.72,
                    ),
                    int((self._phase * 42.0) % 360 * 16),
                    int(28 * 16),
                )

            if effect_profile["consensus"]:
                consensus_pen = QtGui.QPen(QtGui.QColor(theme["success"]))
                consensus_pen.setWidthF(1.0 if self._reduced_effects else 1.6)
                consensus_pen.setStyle(QtCore.Qt.PenStyle.DotLine)
                painter.setPen(consensus_pen)
                painter.drawEllipse(center, radius * 1.88, radius * 1.88)

            if effect_profile["speaking"] and not self._reduced_effects:
                waveform_width = radius * 2.9
                waveform_y = center.y() + (radius * 1.62)
                waveform_pen = QtGui.QPen(QtGui.QColor(primary.red(), primary.green(), primary.blue(), int(120 * intensity)))
                waveform_pen.setWidthF(1.6)
                painter.setPen(waveform_pen)
                path = QtGui.QPainterPath()
                path.moveTo(center.x() - waveform_width / 2.0, waveform_y)
                points = 24 if self._particle_density == "minimal" else 42
                for index in range(points + 1):
                    progress = index / max(1, points)
                    x = center.x() - waveform_width / 2.0 + (waveform_width * progress)
                    wave = math.sin((progress * math.tau * 4.0) + (self._phase * 1.8))
                    amplitude = radius * (0.09 if effect_profile["speaking"] else 0.03)
                    y = waveform_y + (wave * amplitude)
                    path.lineTo(x, y)
                painter.drawPath(path)
            window = self.window()
            record_sample = getattr(window, "_record_perf_sample", None)
            if callable(record_sample):
                record_sample("orb_paint_ms", (time.perf_counter() - started_at) * 1000.0)


    class ShellBackdropWidget(QtWidgets.QWidget):
        """Shared shell backdrop that adds haze and reflection behind the shell frame."""

        def __init__(
            self,
            shell_window: PySideShellWindow | None = None,
            parent: QtWidgets.QWidget | None = None,
        ) -> None:
            super().__init__(parent)
            self._shell_window = shell_window
            self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)

        def _starfield_specs(self, rect: QtCore.QRect, *, reduced_effects: bool, phase: float) -> list[tuple[QtCore.QPointF, float]]:
            shell_window = self._shell_window
            if shell_window is None:
                return []
            ambient_mode = str(shell_window._shell_state.ambient_mode or "steady")
            base_count = 10 if reduced_effects else 18
            if ambient_mode == "deep":
                base_count += 8
            elif ambient_mode == "alert":
                base_count += 4
            specs: list[tuple[QtCore.QPointF, float]] = []
            for index in range(base_count):
                progress = index / max(1, base_count)
                x = rect.width() * (0.12 + (0.76 * ((progress * 1.37) % 1.0)))
                y_band = 0.10 + (0.26 * (((index * 0.23) + (phase * 0.01)) % 1.0))
                y = rect.height() * y_band
                alpha = 0.10 + (0.10 * (0.5 + math.sin((phase * 0.55) + index) * 0.5))
                specs.append((QtCore.QPointF(x, y), alpha))
            return specs

        def _energy_arc_specs(
            self,
            rect: QtCore.QRect,
            *,
            reduced_effects: bool,
            phase: float,
        ) -> list[tuple[QtCore.QRectF, int, int]]:
            shell_window = self._shell_window
            if shell_window is None:
                return []
            ambient_mode = str(shell_window._shell_state.ambient_mode or "steady")
            if ambient_mode == "dormant":
                return []
            arc_count = 1 if reduced_effects else 2 if ambient_mode == "steady" else 3
            specs: list[tuple[QtCore.QRectF, int, int]] = []
            for index in range(arc_count):
                width = rect.width() * (0.44 + (index * 0.10))
                height = rect.height() * (0.12 + (index * 0.04))
                y = rect.height() * (0.22 + (index * 0.05))
                specs.append(
                    (
                        QtCore.QRectF(
                            (rect.width() - width) / 2.0,
                            y,
                            width,
                            height,
                        ),
                        int(((phase * (4.0 + index)) + (index * 28.0)) % 360 * 16),
                        int((150 - (index * 22)) * 16),
                    )
                )
            return specs

        def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # pragma: no cover - exercised via widget tests
            started_at = time.perf_counter()
            del event
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            option = QtWidgets.QStyleOption()
            option.initFrom(self)
            self.style().drawPrimitive(QtWidgets.QStyle.PrimitiveElement.PE_Widget, option, painter, self)
            shell_window = self._shell_window
            if shell_window is None:
                return
            theme = shell_window._current_theme()
            ui = shell_window._ui_preferences
            reduced_motion = bool(ui.get("reduced_motion", False))
            reduced_effects = bool(ui.get("reduced_effects_mode", False) or ui.get("low_resource_mode", False))
            phase = 0.0 if reduced_motion else shell_window._animation_clock.phase
            rect = self.rect()
            accent = _parse_css_color(theme["accent"])
            warning = _parse_css_color(theme["warning"])
            haze_color = warning if shell_window._shell_state.degraded_reason or shell_window._shell_state.fallback_reason else accent
            center = QtCore.QPointF(rect.center().x(), rect.height() * 0.23)
            radius = min(rect.width(), rect.height()) * (0.34 if reduced_effects else 0.40)
            pulse = 1.0 if reduced_motion else 1.0 + (math.sin(phase * 1.1) * 0.04)
            stage_rect = QtCore.QRectF(rect.width() * 0.14, rect.height() * 0.05, rect.width() * 0.72, rect.height() * 0.78)

            if not reduced_effects:
                painter.setPen(QtCore.Qt.PenStyle.NoPen)
                star_color = QtGui.QColor(accent.red(), accent.green(), accent.blue(), 32)
                for point, alpha in self._starfield_specs(rect, reduced_effects=reduced_effects, phase=phase):
                    star_color.setAlpha(max(10, min(60, int(alpha * 255))))
                    painter.setBrush(star_color)
                    painter.drawEllipse(point, 1.4, 1.4)

            cool_glow = QtGui.QRadialGradient(QtCore.QPointF(rect.width() * 0.24, rect.height() * 0.16), rect.width() * 0.38)
            cool_glow.setColorAt(0.0, QtGui.QColor(accent.red(), accent.green(), accent.blue(), 48))
            cool_glow.setColorAt(0.65, QtGui.QColor(accent.red(), accent.green(), accent.blue(), 12))
            cool_glow.setColorAt(1.0, QtGui.QColor(0, 0, 0, 0))
            painter.setBrush(cool_glow)
            painter.drawEllipse(QtCore.QPointF(rect.width() * 0.24, rect.height() * 0.16), rect.width() * 0.34, rect.height() * 0.22)

            warm_glow = QtGui.QRadialGradient(QtCore.QPointF(rect.width() * 0.74, rect.height() * 0.34), rect.width() * 0.40)
            warm_glow.setColorAt(0.0, QtGui.QColor(warning.red(), warning.green(), warning.blue(), 44 if not reduced_effects else 24))
            warm_glow.setColorAt(0.62, QtGui.QColor(warning.red(), warning.green(), warning.blue(), 12))
            warm_glow.setColorAt(1.0, QtGui.QColor(0, 0, 0, 0))
            painter.setBrush(warm_glow)
            painter.drawEllipse(QtCore.QPointF(rect.width() * 0.74, rect.height() * 0.34), rect.width() * 0.32, rect.height() * 0.24)

            haze = QtGui.QRadialGradient(center, radius * pulse)
            haze.setColorAt(0.0, QtGui.QColor(haze_color.red(), haze_color.green(), haze_color.blue(), 68))
            haze.setColorAt(0.55, QtGui.QColor(haze_color.red(), haze_color.green(), haze_color.blue(), 22))
            haze.setColorAt(1.0, QtGui.QColor(0, 0, 0, 0))
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(haze)
            painter.drawEllipse(center, radius, radius * 0.78)

            frame_pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 34 if not reduced_effects else 18))
            frame_pen.setWidthF(1.4)
            painter.setPen(frame_pen)
            frame_gradient = QtGui.QLinearGradient(stage_rect.topLeft(), stage_rect.bottomRight())
            frame_gradient.setColorAt(0.0, QtGui.QColor(accent.red(), accent.green(), accent.blue(), 14))
            frame_gradient.setColorAt(0.55, QtGui.QColor(255, 255, 255, 8))
            frame_gradient.setColorAt(1.0, QtGui.QColor(warning.red(), warning.green(), warning.blue(), 14))
            painter.setBrush(frame_gradient)
            painter.drawRoundedRect(stage_rect, 32.0, 32.0)

            arc_pen = QtGui.QPen(QtGui.QColor(accent.red(), accent.green(), accent.blue(), 36 if not reduced_effects else 18))
            arc_pen.setWidthF(1.2 if reduced_effects else 1.6)
            painter.setPen(arc_pen)
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            for arc_rect, start_angle, span_angle in self._energy_arc_specs(
                rect,
                reduced_effects=reduced_effects,
                phase=phase,
            ):
                painter.drawArc(arc_rect, start_angle, span_angle)

            reflection = QtGui.QLinearGradient(0.0, rect.height() * 0.48, 0.0, rect.height() * 0.82)
            reflection_color = _parse_css_color(theme["reflection"])
            reflection.setColorAt(0.0, QtGui.QColor(reflection_color.red(), reflection_color.green(), reflection_color.blue(), 0))
            reflection.setColorAt(0.35, QtGui.QColor(reflection_color.red(), reflection_color.green(), reflection_color.blue(), 22))
            reflection.setColorAt(1.0, QtGui.QColor(reflection_color.red(), reflection_color.green(), reflection_color.blue(), 0))
            painter.setBrush(reflection)
            painter.drawRoundedRect(
                QtCore.QRectF(rect.width() * 0.12, rect.height() * 0.50, rect.width() * 0.76, rect.height() * 0.28),
                28.0,
                28.0,
            )

            lower_reflection = QtGui.QLinearGradient(0.0, rect.height() * 0.70, 0.0, rect.height() * 0.94)
            lower_reflection.setColorAt(0.0, QtGui.QColor(accent.red(), accent.green(), accent.blue(), 0))
            lower_reflection.setColorAt(0.55, QtGui.QColor(accent.red(), accent.green(), accent.blue(), 18 if not reduced_effects else 10))
            lower_reflection.setColorAt(1.0, QtGui.QColor(accent.red(), accent.green(), accent.blue(), 0))
            painter.setBrush(lower_reflection)
            painter.drawRoundedRect(
                QtCore.QRectF(rect.width() * 0.16, rect.height() * 0.74, rect.width() * 0.68, rect.height() * 0.12),
                20.0,
                20.0,
            )

            line_y = rect.height() * 0.34
            line_shift = 0.0 if reduced_motion else math.sin(phase * 1.5) * (rect.width() * 0.02)
            horizon_pen = QtGui.QPen(QtGui.QColor(accent.red(), accent.green(), accent.blue(), 76 if not reduced_effects else 44))
            horizon_pen.setWidthF(1.4 if reduced_effects else 2.0)
            painter.setPen(horizon_pen)
            painter.drawLine(
                QtCore.QPointF(rect.width() * 0.18 + line_shift, line_y),
                QtCore.QPointF(rect.width() * 0.82 + line_shift, line_y),
            )

            if not reduced_effects:
                waveform_rect = QtCore.QRectF(rect.width() * 0.28, rect.height() * 0.34, rect.width() * 0.44, rect.height() * 0.08)
                for index in range(42):
                    progress = index / 41.0
                    x = waveform_rect.left() + (waveform_rect.width() * progress)
                    amplitude = waveform_rect.height() * (0.16 + (0.84 * abs(math.sin((progress * math.tau * 4.0) + (phase * 2.0)))))
                    if x < rect.center().x():
                        bar_color = QtGui.QColor(accent.red(), accent.green(), accent.blue(), 84)
                    else:
                        bar_color = QtGui.QColor(warning.red(), warning.green(), warning.blue(), 84)
                    painter.setPen(QtGui.QPen(bar_color, 1.6))
                    painter.drawLine(
                        QtCore.QPointF(x, waveform_rect.center().y() - (amplitude * 0.5)),
                        QtCore.QPointF(x, waveform_rect.center().y() + (amplitude * 0.5)),
                    )

            rail_gradient = QtGui.QLinearGradient(rect.width() * 0.08, rect.height() * 0.09, rect.width() * 0.92, rect.height() * 0.09)
            rail_gradient.setColorAt(0.0, QtGui.QColor(accent.red(), accent.green(), accent.blue(), 140))
            rail_gradient.setColorAt(0.55, QtGui.QColor(255, 255, 255, 42))
            rail_gradient.setColorAt(1.0, QtGui.QColor(warning.red(), warning.green(), warning.blue(), 140))
            painter.setPen(QtGui.QPen(QtGui.QBrush(rail_gradient), 2.2))
            painter.drawLine(QtCore.QPointF(rect.width() * 0.08, rect.height() * 0.09), QtCore.QPointF(rect.width() * 0.92, rect.height() * 0.09))

            rail_pen = QtGui.QPen(QtGui.QColor(accent.red(), accent.green(), accent.blue(), 42 if not reduced_effects else 20))
            rail_pen.setWidthF(1.2)
            painter.setPen(rail_pen)
            painter.drawLine(QtCore.QPointF(rect.width() * 0.06, rect.height() * 0.08), QtCore.QPointF(rect.width() * 0.06, rect.height() * 0.92))
            painter.drawLine(QtCore.QPointF(rect.width() * 0.94, rect.height() * 0.08), QtCore.QPointF(rect.width() * 0.94, rect.height() * 0.92))
            shell_window._record_perf_sample(
                "backdrop_paint_ms",
                (time.perf_counter() - started_at) * 1000.0,
            )


    class OperatorSurfacePanel(QtWidgets.QWidget):
        """Reusable operator surface with a summary strip, list, and detail reader."""

        def __init__(self, title: str, parent: QtWidgets.QWidget | None = None) -> None:
            super().__init__(parent)
            layout = QtWidgets.QVBoxLayout(self)
            layout.setContentsMargins(12, 12, 12, 12)
            layout.setSpacing(10)
            header = QtWidgets.QHBoxLayout()
            header.setContentsMargins(0, 0, 0, 0)
            header.setSpacing(8)
            self.kicker_label = QtWidgets.QLabel(_panel_kicker(title), self)
            self.kicker_label.setObjectName("panelKicker")
            self.title_label = QtWidgets.QLabel(title, self)
            header.addWidget(self.kicker_label)
            header.addWidget(self.title_label)
            header.addStretch(1)
            self.summary_label = QtWidgets.QLabel(self)
            self.summary_label.setWordWrap(True)
            self.summary_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            self._splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical, self)
            self._splitter.setChildrenCollapsible(False)
            self._splitter.setHandleWidth(6)
            self.list_widget = QtWidgets.QListWidget(self._splitter)
            self.detail_view = QtWidgets.QPlainTextEdit(self._splitter)
            self.detail_view.setReadOnly(True)
            self._splitter.addWidget(self.list_widget)
            self._splitter.addWidget(self.detail_view)
            self._splitter.setStretchFactor(0, 3)
            self._splitter.setStretchFactor(1, 2)
            self._splitter.setSizes([260, 170])
            layout.addLayout(header)
            layout.addWidget(self.summary_label)
            layout.addWidget(self._splitter, stretch=1)
            self.list_widget.currentItemChanged.connect(self._sync_detail_from_selection)
            self.set_summary("")
            self.set_empty_detail(f"{title} details will appear here when a row is selected.")

        def set_summary(self, text: str) -> None:
            self.summary_label.setText(text)
            self.summary_label.setVisible(bool(str(text).strip()))

        def set_empty_detail(self, text: str) -> None:
            if self.list_widget.currentItem() is None:
                self.detail_view.setPlainText(text)

        def _sync_detail_from_selection(
            self,
            current: QtWidgets.QListWidgetItem | None,
            _previous: QtWidgets.QListWidgetItem | None,
        ) -> None:
            if current is None:
                return
            detail = current.data(_ITEM_DETAIL_ROLE) or current.toolTip() or current.text()
            self.detail_view.setPlainText(str(detail).strip())


    class PySideShellWindow(QtWidgets.QMainWindow):
        """Live-bound premium shell window."""

        def __init__(
            self,
            shell_state: ShellState | None = None,
            app_state: DashboardAppState | None = None,
            *,
            shell_state_provider: Callable[[], ShellState] | None = None,
            app_state_provider: Callable[[], DashboardAppState] | None = None,
            submit_task: Callable[[str, int], bool] | None = None,
            save_settings: Callable[[UserSettingsProfile], bool] | None = None,
            request_action: Callable[[str, dict[str, Any] | None], bool] | None = None,
        ) -> None:
            super().__init__()
            self._shell_state = shell_state or ShellState()
            self._app_state = app_state or DashboardAppState()
            self._shell_state_provider = shell_state_provider
            self._app_state_provider = app_state_provider
            self._submit_task = submit_task
            self._save_settings = save_settings
            self._request_action = request_action
            self._ui_preferences: dict[str, Any] = {}
            self._syncing_dock_visibility = False
            self._dock_manual_visibility: dict[str, bool | None] = {
                "left": None,
                "right": None,
                "bottom": None,
            }
            self._activity_chip_widgets: list[tuple[QtWidgets.QLabel, bool, str]] = []
            self._notification_flash = 0.0
            self._last_notification_id = ""
            self._message_animations: list[QtCore.QAbstractAnimation] = []
            self._conversation_item_ids: tuple[str, ...] = ()
            self._selected_coding_role = ""
            self._suppress_coding_route_signals = False
            self._operator_expanded_mode = False
            self._center_splitter_user_override = False
            self._last_center_splitter_signature: tuple[int, bool] = (0, False)
            self._perf_samples: dict[str, deque[float]] = {
                "orb_paint_ms": deque(maxlen=48),
                "backdrop_paint_ms": deque(maxlen=48),
                "shell_apply_ms": deque(maxlen=48),
                "dashboard_apply_ms": deque(maxlen=48),
                "resize_ms": deque(maxlen=24),
                "dock_sync_ms": deque(maxlen=24),
            }
            self._perf_counters: dict[str, int] = {
                "last_event_burst_items": 0,
                "max_event_burst_items": 0,
                "last_intro_animation_batch": 0,
                "max_intro_animation_batch": 0,
            }
            self.setWindowTitle("Quester.AI")
            self.resize(1600, 980)
            self.setMinimumSize(1200, 780)
            self.setDockOptions(
                self.dockOptions()
                | QtWidgets.QMainWindow.DockOption.AnimatedDocks
                | QtWidgets.QMainWindow.DockOption.AllowTabbedDocks
            )
            self._build_ui()
            initial_theme = _theme(self._shell_state, self._ui_preferences)
            self._display_theme = initial_theme
            self._target_theme = initial_theme
            self._animation_clock = ShellAnimationClock(self)
            self._animation_clock.tick.connect(self._on_animation_tick)
            self._orb.attach_animation_clock(self._animation_clock)
            self._notification_effect = QtWidgets.QGraphicsOpacityEffect(self._notification)
            self._notification.setGraphicsEffect(self._notification_effect)
            self._timer = QtCore.QTimer(self)
            self._timer.timeout.connect(self._poll_state)
            self._timer.start(140)
            if self._shell_state_provider is not None or self._app_state_provider is not None:
                self.refresh_from_state()
            else:
                self.apply_dashboard_state(self._app_state)
                self.apply_shell_state(self._shell_state)
            QtCore.QTimer.singleShot(0, self._prime_workstation_layout)

        def _build_ui(self) -> None:
            central = ShellBackdropWidget(self, self)
            central.setObjectName("shellRoot")
            root = QtWidgets.QVBoxLayout(central)
            root.setContentsMargins(18, 14, 18, 16)
            root.setSpacing(16)
            self._window_chrome = QtWidgets.QFrame(central)
            self._window_chrome.setObjectName("windowChrome")
            chrome_layout = QtWidgets.QHBoxLayout(self._window_chrome)
            chrome_layout.setContentsMargins(18, 12, 18, 12)
            chrome_layout.setSpacing(10)
            self._chrome_app_badge = QtWidgets.QLabel("QUESTER.AI DESKTOP", self._window_chrome)
            self._chrome_app_badge.setObjectName("chromeAppBadge")
            self._chrome_workspace_badge = QtWidgets.QLabel("", self._window_chrome)
            self._chrome_workspace_badge.setObjectName("chromeBadge")
            self._chrome_profile_badge = QtWidgets.QLabel("", self._window_chrome)
            self._chrome_profile_badge.setObjectName("chromeBadge")
            self._chrome_route_badge = QtWidgets.QLabel("", self._window_chrome)
            self._chrome_route_badge.setObjectName("chromeSignalBadge")
            self._chrome_status_badge = QtWidgets.QLabel("", self._window_chrome)
            self._chrome_status_badge.setObjectName("chromeSignalBadge")
            self._chrome_signal_badge = QtWidgets.QLabel("", self._window_chrome)
            self._chrome_signal_badge.setObjectName("chromeSignalBadge")
            chrome_layout.addWidget(self._chrome_app_badge)
            chrome_layout.addWidget(self._chrome_workspace_badge)
            chrome_layout.addWidget(self._chrome_profile_badge)
            chrome_layout.addStretch(1)
            chrome_layout.addWidget(self._chrome_route_badge)
            chrome_layout.addWidget(self._chrome_status_badge)
            chrome_layout.addWidget(self._chrome_signal_badge)
            root.addWidget(self._window_chrome)
            self._hero_surface = QtWidgets.QFrame(central)
            self._hero_surface.setObjectName("heroSurface")
            hero_layout = QtWidgets.QVBoxLayout(self._hero_surface)
            hero_layout.setContentsMargins(22, 18, 22, 20)
            hero_layout.setSpacing(12)
            self._ribbon = QtWidgets.QLabel(self._hero_surface, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
            hero_layout.addWidget(self._ribbon)
            self._orb = OrbWidget(central)
            self._status = QtWidgets.QLabel(alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
            self._sub_status = QtWidgets.QLabel(alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
            self._sub_status.setWordWrap(True)
            self._hero_metrics = QtWidgets.QLabel(alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
            self._hero_metrics.setWordWrap(True)
            self._route_summary = QtWidgets.QLabel(alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
            self._route_summary.setWordWrap(True)
            self._hero_agents = QtWidgets.QLabel(alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
            self._hero_agents.setWordWrap(True)
            self._activity_bar = QtWidgets.QWidget(central)
            self._activity_bar.setObjectName("activityBar")
            self._activity_layout = QtWidgets.QHBoxLayout(self._activity_bar)
            self._activity_layout.setContentsMargins(0, 0, 0, 0)
            self._activity_layout.setSpacing(8)
            self._activity_layout.addStretch(1)
            self._notification = QtWidgets.QLabel(alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
            self._notification.setObjectName("shellNotification")
            self._notification.setWordWrap(True)
            hero_layout.addWidget(self._orb, stretch=5)
            hero_layout.addWidget(self._status)
            hero_layout.addWidget(self._sub_status)
            hero_layout.addWidget(self._hero_metrics)
            hero_layout.addWidget(self._route_summary)
            hero_layout.addWidget(self._hero_agents)
            hero_layout.addWidget(self._activity_bar)
            hero_layout.addWidget(self._notification)
            self._policy_context_bar = QtWidgets.QWidget(central)
            self._policy_context_layout = QtWidgets.QHBoxLayout(self._policy_context_bar)
            self._policy_context_layout.setContentsMargins(0, 0, 0, 0)
            self._policy_context_layout.setSpacing(8)
            self._policy_context_layout.addStretch(1)
            hero_layout.addWidget(self._policy_context_bar)
            self._approval_overlay = QtWidgets.QFrame(central)
            self._approval_overlay.setObjectName("approvalOverlay")
            approval_layout = QtWidgets.QVBoxLayout(self._approval_overlay)
            approval_layout.setContentsMargins(18, 16, 18, 16)
            approval_layout.setSpacing(8)
            approval_header = QtWidgets.QHBoxLayout()
            self._approval_kicker = QtWidgets.QLabel("CONTROL HOLD", self._approval_overlay)
            self._approval_kicker.setObjectName("sectionKicker")
            self._approval_title = QtWidgets.QLabel("Approval Required", self._approval_overlay)
            self._approval_risk = QtWidgets.QLabel("", self._approval_overlay)
            approval_header.addWidget(self._approval_kicker)
            approval_header.addWidget(self._approval_title)
            approval_header.addStretch(1)
            approval_header.addWidget(self._approval_risk)
            approval_layout.addLayout(approval_header)
            self._approval_summary = QtWidgets.QLabel(self._approval_overlay)
            self._approval_summary.setWordWrap(True)
            self._approval_summary.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            approval_layout.addWidget(self._approval_summary)
            self._approval_target = QtWidgets.QLabel(self._approval_overlay)
            self._approval_target.setWordWrap(True)
            self._approval_target.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            approval_layout.addWidget(self._approval_target)
            self._approval_reason = QtWidgets.QLabel(self._approval_overlay)
            self._approval_reason.setWordWrap(True)
            self._approval_reason.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            approval_layout.addWidget(self._approval_reason)
            approval_actions = QtWidgets.QHBoxLayout()
            self._approval_pause_button = QtWidgets.QPushButton("Pause Session", self._approval_overlay)
            self._approval_resume_button = QtWidgets.QPushButton("Resume Session", self._approval_overlay)
            self._approval_stop_button = QtWidgets.QPushButton("Stop Session", self._approval_overlay)
            self._approval_pause_button.clicked.connect(lambda: self._dispatch_session_action("pause"))
            self._approval_resume_button.clicked.connect(lambda: self._dispatch_session_action("resume"))
            self._approval_stop_button.clicked.connect(lambda: self._dispatch_session_action("stop"))
            approval_actions.addWidget(self._approval_pause_button)
            approval_actions.addWidget(self._approval_resume_button)
            approval_actions.addWidget(self._approval_stop_button)
            approval_actions.addStretch(1)
            approval_layout.addLayout(approval_actions)
            hero_layout.addWidget(self._approval_overlay)
            root.addWidget(self._hero_surface, stretch=5)
            self._operations_surface = QtWidgets.QFrame(central)
            self._operations_surface.setObjectName("operationsSurface")
            operations_layout = QtWidgets.QVBoxLayout(self._operations_surface)
            operations_layout.setContentsMargins(18, 16, 18, 16)
            operations_layout.setSpacing(12)
            self._long_horizon_tray = QtWidgets.QFrame(central)
            self._long_horizon_tray.setObjectName("longHorizonTray")
            long_horizon_layout = QtWidgets.QVBoxLayout(self._long_horizon_tray)
            long_horizon_layout.setContentsMargins(18, 14, 18, 14)
            long_horizon_layout.setSpacing(8)
            long_horizon_header = QtWidgets.QHBoxLayout()
            self._long_horizon_kicker = QtWidgets.QLabel("LONG RUN", self._long_horizon_tray)
            self._long_horizon_kicker.setObjectName("sectionKicker")
            self._long_horizon_title = QtWidgets.QLabel("Long-Horizon", self._long_horizon_tray)
            self._long_horizon_phase = QtWidgets.QLabel("", self._long_horizon_tray)
            long_horizon_header.addWidget(self._long_horizon_kicker)
            long_horizon_header.addWidget(self._long_horizon_title)
            long_horizon_header.addStretch(1)
            long_horizon_header.addWidget(self._long_horizon_phase)
            long_horizon_layout.addLayout(long_horizon_header)
            self._long_horizon_summary = QtWidgets.QLabel(self._long_horizon_tray)
            self._long_horizon_summary.setWordWrap(True)
            self._long_horizon_summary.setTextInteractionFlags(
                QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
            )
            long_horizon_layout.addWidget(self._long_horizon_summary)
            self._long_horizon_metrics = QtWidgets.QLabel(self._long_horizon_tray)
            self._long_horizon_metrics.setWordWrap(True)
            self._long_horizon_metrics.setTextInteractionFlags(
                QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
            )
            long_horizon_layout.addWidget(self._long_horizon_metrics)
            self._long_horizon_delta = QtWidgets.QLabel(self._long_horizon_tray)
            self._long_horizon_delta.setWordWrap(True)
            self._long_horizon_delta.setTextInteractionFlags(
                QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
            )
            long_horizon_layout.addWidget(self._long_horizon_delta)
            operations_layout.addWidget(self._long_horizon_tray)
            self._center_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical, central)
            self._center_splitter.setObjectName("centerSplitter")
            self._center_splitter.setChildrenCollapsible(False)
            self._center_splitter.setHandleWidth(8)
            self._center_splitter.splitterMoved.connect(self._on_center_splitter_moved)

            self._summary_scroll = QtWidgets.QScrollArea(self._center_splitter)
            self._summary_scroll.setWidgetResizable(True)
            self._summary_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
            self._summary_container = QtWidgets.QWidget(self._summary_scroll)
            self._summary_container.setObjectName("summarySurface")
            self._summary_layout = QtWidgets.QVBoxLayout(self._summary_container)
            self._summary_layout.setContentsMargins(0, 0, 0, 0)
            self._summary_layout.setSpacing(16)

            self._active_task_card = QtWidgets.QFrame(self._summary_container)
            self._active_task_card.setObjectName("activeTaskCard")
            self._active_task_card.setMinimumHeight(280)
            self._active_task_card.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.MinimumExpanding,
            )
            active_layout = QtWidgets.QVBoxLayout(self._active_task_card)
            active_layout.setContentsMargins(22, 18, 22, 18)
            active_layout.setSpacing(10)
            active_header = QtWidgets.QHBoxLayout()
            self._active_task_kicker = QtWidgets.QLabel("LIVE TASK", self._active_task_card)
            self._active_task_kicker.setObjectName("sectionKicker")
            self._active_task_title = QtWidgets.QLabel("Active Task", self._active_task_card)
            self._active_task_phase = QtWidgets.QLabel("", self._active_task_card)
            active_header.addWidget(self._active_task_kicker)
            active_header.addWidget(self._active_task_title)
            active_header.addStretch(1)
            active_header.addWidget(self._active_task_phase)
            active_layout.addLayout(active_header)
            self._active_task_summary = QtWidgets.QLabel(self._active_task_card)
            self._active_task_summary.setWordWrap(True)
            self._active_task_summary.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            self._active_task_summary.setMinimumHeight(48)
            active_layout.addWidget(self._active_task_summary)
            self._active_task_metrics = QtWidgets.QLabel(self._active_task_card)
            self._active_task_metrics.setWordWrap(True)
            self._active_task_metrics.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            self._active_task_metrics.setMinimumHeight(40)
            active_layout.addWidget(self._active_task_metrics)
            self._active_task_detail_stack = QtWidgets.QWidget(self._active_task_card)
            active_detail_layout = QtWidgets.QGridLayout(self._active_task_detail_stack)
            active_detail_layout.setContentsMargins(0, 0, 0, 0)
            active_detail_layout.setHorizontalSpacing(10)
            active_detail_layout.setVerticalSpacing(10)
            active_detail_layout.setColumnStretch(0, 1)
            active_detail_layout.setColumnStretch(1, 1)

            self._active_route_block = QtWidgets.QFrame(self._active_task_detail_stack)
            self._active_route_block.setObjectName("operatorDetailBlock")
            active_route_layout = QtWidgets.QVBoxLayout(self._active_route_block)
            active_route_layout.setContentsMargins(14, 12, 14, 12)
            active_route_layout.setSpacing(4)
            self._active_route_title = QtWidgets.QLabel("Routing", self._active_route_block)
            self._active_route_title.setObjectName("operatorDetailTitle")
            self._active_task_routes = QtWidgets.QLabel(self._active_route_block)
            self._active_task_routes.setObjectName("operatorDetailValue")
            self._active_task_routes.setWordWrap(True)
            self._active_task_routes.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            active_route_layout.addWidget(self._active_route_title)
            active_route_layout.addWidget(self._active_task_routes)
            active_detail_layout.addWidget(self._active_route_block, 0, 0, 1, 2)

            self._active_roles_block = QtWidgets.QFrame(self._active_task_detail_stack)
            self._active_roles_block.setObjectName("operatorDetailBlock")
            active_roles_layout = QtWidgets.QVBoxLayout(self._active_roles_block)
            active_roles_layout.setContentsMargins(14, 12, 14, 12)
            active_roles_layout.setSpacing(4)
            self._active_roles_title = QtWidgets.QLabel("Roles And Tools", self._active_roles_block)
            self._active_roles_title.setObjectName("operatorDetailTitle")
            self._active_task_roles = QtWidgets.QLabel(self._active_roles_block)
            self._active_task_roles.setObjectName("operatorDetailValue")
            self._active_task_roles.setWordWrap(True)
            self._active_task_roles.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            active_roles_layout.addWidget(self._active_roles_title)
            active_roles_layout.addWidget(self._active_task_roles)
            active_detail_layout.addWidget(self._active_roles_block, 1, 0)

            self._active_warning_block = QtWidgets.QFrame(self._active_task_detail_stack)
            self._active_warning_block.setObjectName("operatorDetailBlock")
            active_warning_layout = QtWidgets.QVBoxLayout(self._active_warning_block)
            active_warning_layout.setContentsMargins(14, 12, 14, 12)
            active_warning_layout.setSpacing(4)
            self._active_warning_title = QtWidgets.QLabel("Operator Signals", self._active_warning_block)
            self._active_warning_title.setObjectName("operatorDetailTitle")
            self._active_task_warnings = QtWidgets.QLabel(self._active_warning_block)
            self._active_task_warnings.setObjectName("operatorDetailValue")
            self._active_task_warnings.setWordWrap(True)
            self._active_task_warnings.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            active_warning_layout.addWidget(self._active_warning_title)
            active_warning_layout.addWidget(self._active_task_warnings)
            active_detail_layout.addWidget(self._active_warning_block, 1, 1)

            active_layout.addWidget(self._active_task_detail_stack, stretch=1)
            self._summary_layout.addWidget(self._active_task_card)

            self._coding_workspace_card = QtWidgets.QFrame(self._summary_container)
            self._coding_workspace_card.setObjectName("codingWorkspaceCard")
            coding_layout = QtWidgets.QVBoxLayout(self._coding_workspace_card)
            coding_layout.setContentsMargins(18, 16, 18, 16)
            coding_layout.setSpacing(8)
            coding_header = QtWidgets.QHBoxLayout()
            self._coding_workspace_kicker = QtWidgets.QLabel("CODE PATH", self._coding_workspace_card)
            self._coding_workspace_kicker.setObjectName("sectionKicker")
            self._coding_workspace_title = QtWidgets.QLabel("Coding Workspace", self._coding_workspace_card)
            self._coding_workspace_status = QtWidgets.QLabel("", self._coding_workspace_card)
            coding_header.addWidget(self._coding_workspace_kicker)
            coding_header.addWidget(self._coding_workspace_title)
            coding_header.addStretch(1)
            coding_header.addWidget(self._coding_workspace_status)
            coding_layout.addLayout(coding_header)
            self._coding_workspace_summary = QtWidgets.QLabel(self._coding_workspace_card)
            self._coding_workspace_summary.setWordWrap(True)
            self._coding_workspace_summary.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            coding_layout.addWidget(self._coding_workspace_summary)
            self._coding_workspace_context = QtWidgets.QLabel(self._coding_workspace_card)
            self._coding_workspace_context.setWordWrap(True)
            self._coding_workspace_context.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            coding_layout.addWidget(self._coding_workspace_context)
            self._coding_workspace_validation = QtWidgets.QLabel(self._coding_workspace_card)
            self._coding_workspace_validation.setWordWrap(True)
            self._coding_workspace_validation.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            coding_layout.addWidget(self._coding_workspace_validation)
            self._coding_workspace_artifacts = QtWidgets.QLabel(self._coding_workspace_card)
            self._coding_workspace_artifacts.setWordWrap(True)
            self._coding_workspace_artifacts.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            coding_layout.addWidget(self._coding_workspace_artifacts)
            self._coding_workspace_blockers = QtWidgets.QLabel(self._coding_workspace_card)
            self._coding_workspace_blockers.setWordWrap(True)
            self._coding_workspace_blockers.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            coding_layout.addWidget(self._coding_workspace_blockers)
            self._summary_layout.addWidget(self._coding_workspace_card)

            self._final_answer_card = QtWidgets.QFrame(self._summary_container)
            self._final_answer_card.setObjectName("finalAnswerCard")
            self._final_answer_card.setMinimumHeight(400)
            self._final_answer_card.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.MinimumExpanding,
            )
            final_layout = QtWidgets.QVBoxLayout(self._final_answer_card)
            final_layout.setContentsMargins(22, 18, 22, 18)
            final_layout.setSpacing(10)
            final_header = QtWidgets.QHBoxLayout()
            self._final_answer_kicker = QtWidgets.QLabel("VERIFIED OUTPUT", self._final_answer_card)
            self._final_answer_kicker.setObjectName("sectionKicker")
            self._final_answer_title = QtWidgets.QLabel("Final Answer", self._final_answer_card)
            self._final_answer_status = QtWidgets.QLabel("", self._final_answer_card)
            final_header.addWidget(self._final_answer_kicker)
            final_header.addWidget(self._final_answer_title)
            final_header.addStretch(1)
            final_header.addWidget(self._final_answer_status)
            final_layout.addLayout(final_header)
            self._final_answer_body = QtWidgets.QLabel(self._final_answer_card)
            self._final_answer_body.setWordWrap(True)
            self._final_answer_body.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            self._final_answer_body.setMinimumHeight(84)
            final_layout.addWidget(self._final_answer_body)
            self._final_answer_evidence = QtWidgets.QLabel(self._final_answer_card)
            self._final_answer_evidence.setWordWrap(True)
            self._final_answer_evidence.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            final_layout.addWidget(self._final_answer_evidence)
            self._final_answer_refs = QtWidgets.QLabel(self._final_answer_card)
            self._final_answer_refs.setWordWrap(True)
            self._final_answer_refs.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            final_layout.addWidget(self._final_answer_refs)
            self._final_answer_warnings = QtWidgets.QLabel(self._final_answer_card)
            self._final_answer_warnings.setWordWrap(True)
            self._final_answer_warnings.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            final_layout.addWidget(self._final_answer_warnings)
            self._final_answer_sections = QtWidgets.QToolBox(self._final_answer_card)
            self._final_answer_sections.setObjectName("finalAnswerSections")
            self._final_answer_sections.setMinimumHeight(170)
            self._final_answer_sections.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Expanding,
            )
            self._why_answer_text = QtWidgets.QPlainTextEdit(self._final_answer_sections)
            self._why_answer_text.setReadOnly(True)
            self._why_answer_text.setMinimumHeight(140)
            self._how_verified_text = QtWidgets.QPlainTextEdit(self._final_answer_sections)
            self._how_verified_text.setReadOnly(True)
            self._how_verified_text.setMinimumHeight(140)
            self._deep_mode_text = QtWidgets.QPlainTextEdit(self._final_answer_sections)
            self._deep_mode_text.setReadOnly(True)
            self._deep_mode_text.setMinimumHeight(140)
            self._final_answer_sections.addItem(self._why_answer_text, "Why This Output")
            self._final_answer_sections.addItem(self._how_verified_text, "How It Was Verified")
            self._final_answer_sections.addItem(self._deep_mode_text, "What Deep Mode Changed")
            final_layout.addWidget(self._final_answer_sections, stretch=1)
            final_actions = QtWidgets.QHBoxLayout()
            self._copy_answer_button = QtWidgets.QPushButton("Copy Answer", self._final_answer_card)
            self._copy_citations_button = QtWidgets.QPushButton("Copy Citations", self._final_answer_card)
            self._copy_answer_button.clicked.connect(self._copy_final_answer)
            self._copy_citations_button.clicked.connect(self._copy_final_references)
            final_actions.addWidget(self._copy_answer_button)
            final_actions.addWidget(self._copy_citations_button)
            final_actions.addStretch(1)
            final_layout.addLayout(final_actions)
            self._summary_layout.addWidget(self._final_answer_card)
            self._summary_layout.addStretch(1)
            self._summary_scroll.setWidget(self._summary_container)

            self._conversation_panel = QtWidgets.QFrame(self._center_splitter)
            self._conversation_panel.setObjectName("conversationArchivePanel")
            self._conversation_panel.setMinimumHeight(180)
            conversation_panel_layout = QtWidgets.QVBoxLayout(self._conversation_panel)
            conversation_panel_layout.setContentsMargins(18, 16, 18, 16)
            conversation_panel_layout.setSpacing(12)
            conversation_panel_header = QtWidgets.QHBoxLayout()
            self._conversation_kicker = QtWidgets.QLabel("ARCHIVE", self._conversation_panel)
            self._conversation_kicker.setObjectName("sectionKicker")
            self._conversation_heading = QtWidgets.QLabel("Conversation Archive", self._conversation_panel)
            self._conversation_hint = QtWidgets.QLabel("Older messages stay secondary to the live task surface.", self._conversation_panel)
            conversation_panel_header.addWidget(self._conversation_kicker)
            conversation_panel_header.addWidget(self._conversation_heading)
            conversation_panel_header.addStretch(1)
            conversation_panel_header.addWidget(self._conversation_hint)
            conversation_panel_layout.addLayout(conversation_panel_header)
            self._conversation_scroll = QtWidgets.QScrollArea(self._conversation_panel)
            self._conversation_scroll.setWidgetResizable(True)
            self._conversation_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
            self._conversation_container = QtWidgets.QWidget(self._conversation_scroll)
            self._conversation_container.setObjectName("conversationSurface")
            self._conversation_layout = QtWidgets.QVBoxLayout(self._conversation_container)
            self._conversation_layout.setContentsMargins(0, 0, 0, 0)
            self._conversation_layout.setSpacing(14)
            self._conversation_stream = QtWidgets.QWidget(self._conversation_container)
            self._conversation_stream_layout = QtWidgets.QVBoxLayout(self._conversation_stream)
            self._conversation_stream_layout.setContentsMargins(0, 0, 0, 0)
            self._conversation_stream_layout.setSpacing(14)
            self._conversation_layout.addWidget(self._conversation_stream)
            self._conversation_scroll.setWidget(self._conversation_container)
            conversation_panel_layout.addWidget(self._conversation_scroll, stretch=1)
            self._center_splitter.addWidget(self._summary_scroll)
            self._center_splitter.addWidget(self._conversation_panel)
            self._center_splitter.setStretchFactor(0, 4)
            self._center_splitter.setStretchFactor(1, 3)
            operations_layout.addWidget(self._center_splitter, stretch=4)
            dock = QtWidgets.QFrame(central)
            dock.setObjectName("controlDock")
            dock_layout = QtWidgets.QVBoxLayout(dock)
            dock_layout.setContentsMargins(18, 16, 18, 16)
            dock_layout.setSpacing(8)
            top = QtWidgets.QHBoxLayout()
            self._mic = QtWidgets.QPushButton("Mic", dock)
            self._mic.setEnabled(False)
            self._input = QtWidgets.QLineEdit(dock)
            self._input.setMinimumHeight(48)
            self._input.setPlaceholderText("Type a message...")
            self._input.returnPressed.connect(self._submit_from_input)
            self._send = QtWidgets.QPushButton("Send", dock)
            self._send.setMinimumHeight(46)
            self._send.clicked.connect(self._submit_from_input)
            top.addWidget(self._mic)
            top.addWidget(self._input, stretch=1)
            top.addWidget(self._send)
            bottom = QtWidgets.QHBoxLayout()
            self._thinking = QtWidgets.QLabel("Thinking: 30 min", dock)
            self._slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, dock)
            self._slider.setRange(1, 720)
            self._slider.setValue(30)
            self._slider.valueChanged.connect(lambda: self._thinking.setText(f"Thinking: {self._slider.value()} min"))
            self._slider.sliderReleased.connect(self._sync_slider_preference)
            self._fast = QtWidgets.QPushButton("Fast", dock)
            self._fast.setCheckable(True)
            self._deep = QtWidgets.QPushButton("Deep", dock)
            self._deep.setCheckable(True)
            self._long = QtWidgets.QPushButton("Long Horizon", dock)
            self._long.setCheckable(True)
            self._pause = QtWidgets.QPushButton("Pause", dock)
            self._resume = QtWidgets.QPushButton("Resume", dock)
            self._stop = QtWidgets.QPushButton("Stop", dock)
            self._dock_status = QtWidgets.QLabel("", dock)
            self._fast.clicked.connect(lambda: self._set_reasoning_preset("fast", 5))
            self._deep.clicked.connect(lambda: self._set_reasoning_preset("deep", 30))
            self._long.clicked.connect(lambda: self._set_reasoning_preset("long_horizon", 180))
            self._pause.clicked.connect(lambda: self._dispatch_session_action("pause"))
            self._resume.clicked.connect(lambda: self._dispatch_session_action("resume"))
            self._stop.clicked.connect(lambda: self._dispatch_session_action("stop"))
            for widget in (
                self._thinking,
                self._slider,
                self._fast,
                self._deep,
                self._long,
                self._pause,
                self._resume,
                self._stop,
                self._dock_status,
            ):
                bottom.addWidget(widget, 1 if widget is self._slider else 0)
            toggle_row = QtWidgets.QHBoxLayout()
            self._local_only_toggle = QtWidgets.QPushButton("Local-Only", dock)
            self._local_only_toggle.setCheckable(True)
            self._local_only_toggle.clicked.connect(
                lambda checked: self._toggle_local_only(bool(checked))
            )
            self._web_toggle = QtWidgets.QPushButton("Web", dock)
            self._web_toggle.setCheckable(True)
            self._web_toggle.clicked.connect(lambda checked: self._toggle_web_access(bool(checked)))
            self._verify_toggle = QtWidgets.QPushButton("Verification", dock)
            self._verify_toggle.setCheckable(True)
            self._verify_toggle.clicked.connect(
                lambda checked: self._toggle_verification_priority(bool(checked))
            )
            self._capability_toggle = QtWidgets.QPushButton("Capability", dock)
            self._capability_toggle.setCheckable(True)
            self._capability_toggle.clicked.connect(
                lambda checked: self._toggle_capability_session(bool(checked))
            )
            self._cloud_toggle = QtWidgets.QPushButton("Cloud Helper", dock)
            self._cloud_toggle.setCheckable(True)
            self._cloud_toggle.clicked.connect(
                lambda checked: self._toggle_cloud_helper(bool(checked))
            )
            for widget in (
                self._local_only_toggle,
                self._web_toggle,
                self._verify_toggle,
                self._capability_toggle,
                self._cloud_toggle,
            ):
                toggle_row.addWidget(widget)
            toggle_row.addStretch(1)
            utility_row = QtWidgets.QHBoxLayout()
            self._timeline_shortcut = QtWidgets.QPushButton("Timeline", dock)
            self._timeline_shortcut.clicked.connect(lambda: self.open_surface("timeline"))
            self._session_shortcut = QtWidgets.QPushButton("Session", dock)
            self._session_shortcut.clicked.connect(lambda: self.open_surface("session"))
            self._insights_shortcut = QtWidgets.QPushButton("Insights", dock)
            self._insights_shortcut.clicked.connect(lambda: self.open_surface("evidence"))
            self._history_shortcut = QtWidgets.QPushButton("History", dock)
            self._history_shortcut.clicked.connect(lambda: self._show_bottom_tab(0))
            self._settings_shortcut = QtWidgets.QPushButton("Settings", dock)
            self._settings_shortcut.clicked.connect(lambda: self._show_bottom_tab(2))
            self._readiness_shortcut = QtWidgets.QPushButton("Readiness", dock)
            self._readiness_shortcut.clicked.connect(self._refresh_readiness_surfaces)
            self._support_bundle_button = QtWidgets.QPushButton("Support Bundle", dock)
            self._support_bundle_button.clicked.connect(self._export_support_bundle)
            self._trace_export_button = QtWidgets.QPushButton("Export Timeline", dock)
            self._trace_export_button.clicked.connect(self._export_selected_trace)
            for widget in (
                self._timeline_shortcut,
                self._session_shortcut,
                self._insights_shortcut,
                self._history_shortcut,
                self._settings_shortcut,
                self._readiness_shortcut,
                self._support_bundle_button,
                self._trace_export_button,
            ):
                utility_row.addWidget(widget)
            utility_row.addStretch(1)
            self._coding_dock_bar = QtWidgets.QWidget(dock)
            self._coding_dock_layout = QtWidgets.QHBoxLayout(self._coding_dock_bar)
            self._coding_dock_layout.setContentsMargins(0, 0, 0, 0)
            self._coding_dock_layout.setSpacing(8)
            self._coding_dock_layout.addStretch(1)
            dock_layout.addLayout(top)
            dock_layout.addLayout(bottom)
            dock_layout.addLayout(toggle_row)
            dock_layout.addLayout(utility_row)
            dock_layout.addWidget(self._coding_dock_bar)
            operations_layout.addWidget(dock)
            root.addWidget(self._operations_surface, stretch=4)
            self.setCentralWidget(central)

            self._agent_list = QtWidgets.QListWidget()
            self._agent_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
            self._timeline_panel = OperatorSurfacePanel("Current Task Timeline", self)
            self._timeline = self._timeline_panel.list_widget
            self._session_panel = QtWidgets.QWidget()
            session_layout = QtWidgets.QVBoxLayout(self._session_panel)
            session_layout.setContentsMargins(12, 12, 12, 12)
            session_layout.setSpacing(10)
            self._session_summary = QtWidgets.QLabel(self._session_panel)
            self._session_summary.setWordWrap(True)
            self._session_summary.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            session_layout.addWidget(self._session_summary)
            self._session_badges = QtWidgets.QWidget(self._session_panel)
            self._session_badges_layout = QtWidgets.QHBoxLayout(self._session_badges)
            self._session_badges_layout.setContentsMargins(0, 0, 0, 0)
            self._session_badges_layout.setSpacing(8)
            self._session_badges_layout.addStretch(1)
            session_layout.addWidget(self._session_badges)
            session_button_row = QtWidgets.QHBoxLayout()
            self._session_pause_button = QtWidgets.QPushButton("Pause", self._session_panel)
            self._session_resume_button = QtWidgets.QPushButton("Resume", self._session_panel)
            self._session_stop_button = QtWidgets.QPushButton("Stop", self._session_panel)
            self._session_kill_button = QtWidgets.QPushButton("Kill", self._session_panel)
            self._session_pause_button.clicked.connect(lambda: self._dispatch_session_action("pause"))
            self._session_resume_button.clicked.connect(lambda: self._dispatch_session_action("resume"))
            self._session_stop_button.clicked.connect(lambda: self._dispatch_session_action("stop"))
            self._session_kill_button.clicked.connect(self._dispatch_kill_action)
            session_button_row.addWidget(self._session_pause_button)
            session_button_row.addWidget(self._session_resume_button)
            session_button_row.addWidget(self._session_stop_button)
            session_button_row.addWidget(self._session_kill_button)
            session_button_row.addStretch(1)
            session_layout.addLayout(session_button_row)
            self._session_diagnostics = QtWidgets.QListWidget(self._session_panel)
            self._session_diagnostics.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
            session_layout.addWidget(self._session_diagnostics)
            self._left_tabs = QtWidgets.QTabWidget(self)
            self._left_tabs.setObjectName("operatorTabStack")
            self._left_tabs.setDocumentMode(True)
            self._left_tabs.tabBar().setExpanding(True)
            self._left_tabs.addTab(self._agent_list, "Agents")
            self._left_tabs.addTab(self._timeline_panel, "Timeline")
            self._left_tabs.addTab(self._session_panel, "Session")
            self._left_dock = QtWidgets.QDockWidget("Task Timeline", self)
            self._left_dock.setObjectName("leftOperatorDock")
            self._left_dock.setFeatures(QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable)
            self._left_dock.setMinimumWidth(292)
            self._left_dock.setMaximumWidth(350)
            self._left_dock.setWidget(self._left_tabs)
            self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, self._left_dock)
            self._left_dock.visibilityChanged.connect(
                lambda visible: self._on_dock_visibility_changed("left", bool(visible))
            )

            self._evidence_panel = OperatorSurfacePanel("Evidence Inspector", self)
            self._evidence_list = self._evidence_panel.list_widget
            self._evidence_summary = self._evidence_panel.summary_label
            self._evidence_detail = self._evidence_panel.detail_view
            self._provenance_panel = OperatorSurfacePanel("Provenance Inspector", self)
            self._provenance_list = self._provenance_panel.list_widget
            self._provenance_summary = self._provenance_panel.summary_label
            self._provenance_detail = self._provenance_panel.detail_view
            self._compression_panel = OperatorSurfacePanel("Compressor Insights", self)
            self._compression_list = self._compression_panel.list_widget
            self._compression_summary = self._compression_panel.summary_label
            self._compression_detail = self._compression_panel.detail_view
            self._optimizer_panel = OperatorSurfacePanel("Optimizer Advisory", self)
            self._optimizer_list = self._optimizer_panel.list_widget
            self._optimizer_summary = self._optimizer_panel.summary_label
            self._optimizer_detail = self._optimizer_panel.detail_view
            self._control_plane_panel = OperatorSurfacePanel("Local AI Control Plane", self)
            self._control_plane_list = self._control_plane_panel.list_widget
            self._control_plane_summary = self._control_plane_panel.summary_label
            self._control_plane_detail = self._control_plane_panel.detail_view
            self._runtime_panel = OperatorSurfacePanel("Runtime Health", self)
            self._runtime_list = self._runtime_panel.list_widget
            self._runtime_summary = self._runtime_panel.summary_label
            self._runtime_detail = self._runtime_panel.detail_view
            self._practice_log_panel = OperatorSurfacePanel("Idle Practice Log", self)
            self._practice_log_list = self._practice_log_panel.list_widget
            self._practice_log_summary = self._practice_log_panel.summary_label
            self._practice_log_detail = self._practice_log_panel.detail_view
            self._coding_patterns_panel = OperatorSurfacePanel("Indexed Coding Patterns", self)
            self._coding_patterns_list = self._coding_patterns_panel.list_widget
            self._coding_patterns_summary = self._coding_patterns_panel.summary_label
            self._coding_patterns_detail = self._coding_patterns_panel.detail_view
            self._coding_memory_panel = OperatorSurfacePanel("Coding Memory Tiers", self)
            self._coding_memory_list = self._coding_memory_panel.list_widget
            self._coding_memory_summary = self._coding_memory_panel.summary_label
            self._coding_memory_detail = self._coding_memory_panel.detail_view
            self._coding_validation_panel = OperatorSurfacePanel("Validation History", self)
            self._coding_validation_list = self._coding_validation_panel.list_widget
            self._coding_validation_summary = self._coding_validation_panel.summary_label
            self._coding_validation_detail = self._coding_validation_panel.detail_view
            self._coding_metrics_panel = OperatorSurfacePanel("Coding Metrics", self)
            self._coding_metrics_list = self._coding_metrics_panel.list_widget
            self._coding_metrics_summary = self._coding_metrics_panel.summary_label
            self._coding_metrics_detail = self._coding_metrics_panel.detail_view
            self._coding_routes_panel = QtWidgets.QWidget()
            coding_routes_layout = QtWidgets.QVBoxLayout(self._coding_routes_panel)
            coding_routes_layout.setContentsMargins(12, 12, 12, 12)
            coding_routes_layout.setSpacing(10)
            self._coding_route_summary = QtWidgets.QLabel(self._coding_routes_panel)
            self._coding_route_summary.setWordWrap(True)
            self._coding_route_summary.setTextInteractionFlags(
                QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
            )
            coding_routes_layout.addWidget(self._coding_route_summary)
            coding_route_switch_row = QtWidgets.QHBoxLayout()
            self._coding_role_combo = QtWidgets.QComboBox(self._coding_routes_panel)
            self._coding_model_combo = QtWidgets.QComboBox(self._coding_routes_panel)
            self._coding_role_combo.currentIndexChanged.connect(self._on_coding_role_changed)
            self._coding_model_combo.currentIndexChanged.connect(self._on_coding_model_changed)
            coding_route_switch_row.addWidget(self._coding_role_combo, 1)
            coding_route_switch_row.addWidget(self._coding_model_combo, 2)
            coding_routes_layout.addLayout(coding_route_switch_row)
            self._coding_route_note = QtWidgets.QLabel(self._coding_routes_panel)
            self._coding_route_note.setWordWrap(True)
            self._coding_route_note.setTextInteractionFlags(
                QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
            )
            coding_routes_layout.addWidget(self._coding_route_note)
            self._coding_routes_list = QtWidgets.QListWidget(self._coding_routes_panel)
            coding_routes_layout.addWidget(self._coding_routes_list)
            self._right_tabs = QtWidgets.QTabWidget(self)
            self._right_tabs.setObjectName("operatorTabStack")
            self._right_tabs.setDocumentMode(True)
            self._right_tabs.tabBar().setUsesScrollButtons(True)
            self._right_tabs.addTab(self._evidence_panel, "Evidence")
            self._right_tabs.addTab(self._provenance_panel, "Provenance")
            self._right_tabs.addTab(self._compression_panel, "Compressor")
            self._right_tabs.addTab(self._optimizer_panel, "Optimizer")
            self._right_tabs.addTab(self._control_plane_panel, "Control Plane")
            self._right_tabs.addTab(self._runtime_panel, "Runtime")
            self._right_tabs.addTab(self._practice_log_panel, "Practice")
            self._right_tabs.addTab(self._coding_patterns_panel, "Patterns")
            self._right_tabs.addTab(self._coding_memory_panel, "Memory")
            self._right_tabs.addTab(self._coding_validation_panel, "Validation")
            self._right_tabs.addTab(self._coding_metrics_panel, "Metrics")
            self._right_tabs.addTab(self._coding_routes_panel, "Coding Routes")
            self._right_dock = QtWidgets.QDockWidget("Evidence And Insights", self)
            self._right_dock.setObjectName("rightOperatorDock")
            self._right_dock.setFeatures(QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable)
            self._right_dock.setMinimumWidth(308)
            self._right_dock.setMaximumWidth(370)
            self._right_dock.setWidget(self._right_tabs)
            self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self._right_dock)
            self._right_dock.visibilityChanged.connect(
                lambda visible: self._on_dock_visibility_changed("right", bool(visible))
            )

            self._history_panel = QtWidgets.QWidget()
            history_layout = QtWidgets.QVBoxLayout(self._history_panel)
            history_layout.setContentsMargins(12, 12, 12, 12)
            history_layout.setSpacing(10)
            self._history_list = QtWidgets.QListWidget(self._history_panel)
            self._history_list.currentRowChanged.connect(self._refresh_history_details)
            history_layout.addWidget(self._history_list)
            self._history_detail = QtWidgets.QPlainTextEdit(self._history_panel)
            self._history_detail.setReadOnly(True)
            history_layout.addWidget(self._history_detail)
            self._knowledge_list = QtWidgets.QListWidget()
            self._settings = QtWidgets.QPlainTextEdit(readOnly=True)
            self._readiness = QtWidgets.QPlainTextEdit(readOnly=True)
            self._capability = QtWidgets.QPlainTextEdit(readOnly=True)
            self._debug = QtWidgets.QPlainTextEdit(readOnly=True)
            self._bottom_tabs = QtWidgets.QTabWidget(self)
            self._bottom_tabs.setObjectName("utilityTabStack")
            self._bottom_tabs.setDocumentMode(True)
            self._bottom_tabs.tabBar().setUsesScrollButtons(True)
            for name, widget in (("History", self._history_panel), ("Knowledge", self._knowledge_list), ("Settings", self._settings), ("Readiness", self._readiness), ("Capabilities", self._capability), ("Debug", self._debug)):
                self._bottom_tabs.addTab(widget, name)
            self._bottom_dock = QtWidgets.QDockWidget("Secondary Views", self)
            self._bottom_dock.setObjectName("bottomOperatorDock")
            self._bottom_dock.setFeatures(QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable)
            self._bottom_dock.setMinimumHeight(210)
            self._bottom_dock.setMaximumHeight(320)
            self._bottom_dock.setWidget(self._bottom_tabs)
            self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, self._bottom_dock)
            self._bottom_dock.visibilityChanged.connect(
                lambda visible: self._on_dock_visibility_changed("bottom", bool(visible))
            )
            self._sync_dock_visibility(reset_manual=True)
            QtGui.QShortcut(QtGui.QKeySequence("Ctrl+L"), self, activated=self._input.setFocus)
            QtGui.QShortcut(QtGui.QKeySequence("Ctrl+1"), self, activated=lambda: self._show_bottom_tab(0))
            QtGui.QShortcut(QtGui.QKeySequence("Ctrl+2"), self, activated=lambda: self._show_bottom_tab(2))
            QtGui.QShortcut(QtGui.QKeySequence("Ctrl+3"), self, activated=self._refresh_readiness_surfaces)
            QtGui.QShortcut(QtGui.QKeySequence("Ctrl+4"), self, activated=lambda: self.open_surface("timeline"))
            QtGui.QShortcut(QtGui.QKeySequence("Ctrl+5"), self, activated=lambda: self.open_surface("evidence"))

        def _poll_state(self) -> None:
            self.refresh_from_state()

        def _dispatch_session_action(self, action_name: str) -> None:
            if self._request_action is None:
                return
            session = self._app_state.local_task_session
            task = self._app_state.active_task
            if action_name == "pause":
                if session.session_id:
                    self._request_action("local_task_session.pause", {"session_id": session.session_id})
                elif task.long_horizon_session_id:
                    self._request_action("long_horizon.pause", {"session_id": task.long_horizon_session_id})
            elif action_name == "resume":
                if session.session_id:
                    self._request_action("local_task_session.resume", {"session_id": session.session_id})
                elif task.long_horizon_session_id:
                    self._request_action("long_horizon.resume", {"session_id": task.long_horizon_session_id})
            elif session.session_id:
                self._request_action("local_task_session.stop", {"session_id": session.session_id})
            elif task.long_horizon_session_id:
                self._request_action("long_horizon.cancel", {"session_id": task.long_horizon_session_id})

        def _submit_from_input(self) -> None:
            question = self._input.text().strip()
            if question and self._submit_task is not None and self._submit_task(question, int(self._slider.value())):
                self._input.clear()

        def _dispatch_kill_action(self) -> None:
            if self._request_action is None:
                return
            session = self._app_state.local_task_session
            if session.session_id:
                self._request_action("local_task_session.kill", {"session_id": session.session_id})

        def _show_left_tab(self, index: int) -> None:
            self._dock_manual_visibility["left"] = True
            self._sync_dock_visibility()
            self._left_tabs.setCurrentIndex(max(0, min(self._left_tabs.count() - 1, int(index))))

        def _show_right_tab(self, index: int) -> None:
            self._dock_manual_visibility["right"] = True
            self._sync_dock_visibility()
            self._right_tabs.setCurrentIndex(max(0, min(self._right_tabs.count() - 1, int(index))))

        def _show_bottom_tab(self, index: int) -> None:
            self._dock_manual_visibility["bottom"] = True
            self._sync_dock_visibility()
            self._bottom_tabs.setCurrentIndex(max(0, min(self._bottom_tabs.count() - 1, int(index))))

        def _dock_default_visibility(self, dock_name: str) -> bool:
            width = self.width()
            if dock_name == "left":
                return bool(self._ui_preferences.get("left_drawer_visible", True)) and width >= _LEFT_DOCK_WIDE_THRESHOLD
            if dock_name == "right":
                return bool(self._ui_preferences.get("right_drawer_visible", True)) and width >= _RIGHT_DOCK_WIDE_THRESHOLD
            if dock_name == "bottom":
                return bool(self._ui_preferences.get("show_utility_drawer", False)) and width >= _BOTTOM_DOCK_WIDE_THRESHOLD
            return False

        def _desired_dock_visibility(self, dock_name: str) -> bool:
            manual = self._dock_manual_visibility.get(dock_name)
            if manual is not None:
                return bool(manual)
            if not bool(self._ui_preferences.get("adaptive_drawers", True)):
                if dock_name == "left":
                    return bool(self._ui_preferences.get("left_drawer_visible", True))
                if dock_name == "right":
                    return bool(self._ui_preferences.get("right_drawer_visible", True))
                if dock_name == "bottom":
                    return bool(self._ui_preferences.get("show_utility_drawer", False))
                return False
            return self._dock_default_visibility(dock_name)

        def _sync_dock_visibility(self, *, reset_manual: bool = False) -> None:
            started_at = time.perf_counter()
            if reset_manual:
                self._dock_manual_visibility = {"left": None, "right": None, "bottom": None}
            self._syncing_dock_visibility = True
            try:
                self._left_dock.setVisible(self._desired_dock_visibility("left"))
                self._right_dock.setVisible(self._desired_dock_visibility("right"))
                self._bottom_dock.setVisible(self._desired_dock_visibility("bottom"))
            finally:
                self._syncing_dock_visibility = False
                self._record_perf_sample("dock_sync_ms", (time.perf_counter() - started_at) * 1000.0)

        def _on_dock_visibility_changed(self, dock_name: str, visible: bool) -> None:
            if self._syncing_dock_visibility:
                return
            if self._dock_manual_visibility.get(dock_name) is None and bool(visible) == self._dock_default_visibility(
                dock_name
            ):
                return
            self._dock_manual_visibility[dock_name] = bool(visible)

        def _visible_summary_card_count(self) -> int:
            return sum(
                1
                for widget in (self._active_task_card, self._coding_workspace_card, self._final_answer_card)
                if not widget.isHidden()
            )

        @QtCore.Slot(int, int)
        def _on_center_splitter_moved(self, _pos: int, _index: int) -> None:
            self._center_splitter_user_override = True

        @QtCore.Slot()
        def _prime_workstation_layout(self) -> None:
            self.resizeDocks(
                [self._left_dock, self._right_dock],
                [308, 332],
                QtCore.Qt.Orientation.Horizontal,
            )
            self.resizeDocks([self._bottom_dock], [248], QtCore.Qt.Orientation.Vertical)
            self._update_center_splitter_layout(force=True)

        def _update_center_splitter_layout(self, *, force: bool = False) -> None:
            visible_count = self._visible_summary_card_count()
            signature = (visible_count, self._operator_expanded_mode)
            signature_changed = signature != self._last_center_splitter_signature
            if signature_changed:
                self._last_center_splitter_signature = signature
            if visible_count <= 0:
                self._summary_scroll.setVisible(False)
                self._center_splitter.setSizes([0, max(220, self._center_splitter.height() or 420)])
                return
            self._summary_scroll.setVisible(True)
            if self._center_splitter_user_override and not force and not signature_changed:
                return
            available_height = max(420, self._center_splitter.height() or (self.height() - 360))
            if self._operator_expanded_mode:
                summary_ratio = 0.78 if visible_count >= 2 else 0.66
            elif visible_count >= 2:
                summary_ratio = 0.70
            else:
                summary_ratio = 0.54
            summary_height = max(260, int(available_height * summary_ratio))
            conversation_height = max(150, available_height - summary_height)
            self._center_splitter.setSizes([summary_height, conversation_height])
            if signature_changed:
                self._center_splitter_user_override = False

        def _default_export_path(self, stem: str, suffix: str) -> str:
            safe_profile = (
                str(self._app_state.user_settings.profile_name or "default")
                .strip()
                .replace(" ", "_")
                .replace("/", "_")
            )
            return str(Path("logs") / "shell_exports" / f"{safe_profile}_{stem}{suffix}")

        @QtCore.Slot()
        def refresh_from_state(self) -> None:
            if self._app_state_provider is not None:
                self.apply_dashboard_state(self._app_state_provider())
            if self._shell_state_provider is not None:
                self.apply_shell_state(self._shell_state_provider())

        def open_surface(self, surface_id: str) -> bool:
            normalized = str(surface_id).strip().lower()
            if not normalized:
                return False
            left_surfaces = {"left_drawer": None, "agents": 0, "timeline": 1, "session": 2}
            right_surfaces = {
                "right_drawer": None,
                "evidence": 0,
                "provenance": 1,
                "compressor": 2,
                "optimizer": 3,
                "control_plane": 4,
                "runtime": 5,
                "practice": 6,
                "patterns": 7,
                "coding_memory": 8,
                "memory": 8,
                "validation": 9,
                "coding_metrics": 10,
                "metrics": 10,
                "coding_routes": 11,
            }
            bottom_surfaces = {
                "utility_drawer": None,
                "history": 0,
                "knowledge": 1,
                "settings": 2,
                "readiness": 3,
                "capabilities": 4,
                "debug": 5,
            }
            if normalized in left_surfaces:
                target_index = left_surfaces[normalized]
                self._show_left_tab(self._left_tabs.currentIndex() if target_index is None else target_index)
                return True
            if normalized in right_surfaces:
                target_index = right_surfaces[normalized]
                self._show_right_tab(self._right_tabs.currentIndex() if target_index is None else target_index)
                return True
            if normalized in bottom_surfaces:
                target_index = bottom_surfaces[normalized]
                self._show_bottom_tab(self._bottom_tabs.currentIndex() if target_index is None else target_index)
                return True
            return False

        @QtCore.Slot(str)
        def _open_surface_slot(self, surface_id: str) -> None:
            self.open_surface(surface_id)

        def _animate_widget_intro(self, widget: QtWidgets.QWidget) -> None:
            if bool(self._ui_preferences.get("reduced_motion", False)):
                return
            effect = QtWidgets.QGraphicsOpacityEffect(widget)
            widget.setGraphicsEffect(effect)
            animation = QtCore.QPropertyAnimation(effect, b"opacity", widget)
            animation.setDuration(260 if not bool(self._ui_preferences.get("reduced_effects_mode", False)) else 140)
            animation.setStartValue(0.0)
            animation.setEndValue(1.0)
            animation.finished.connect(
                lambda effect=effect, widget=widget, animation=animation: self._finalize_intro_animation(
                    widget,
                    effect,
                    animation,
                )
            )
            self._message_animations.append(animation)
            animation.start()

        def _finalize_intro_animation(
            self,
            widget: QtWidgets.QWidget,
            effect: QtWidgets.QGraphicsOpacityEffect,
            animation: QtCore.QAbstractAnimation,
        ) -> None:
            if widget.graphicsEffect() is effect:
                widget.setGraphicsEffect(None)
            if animation in self._message_animations:
                self._message_animations.remove(animation)

        def _bounded_autoscroll(self, *, force: bool = False) -> None:
            scrollbar = self._conversation_scroll.verticalScrollBar()
            near_bottom = force or (scrollbar.maximum() - scrollbar.value()) <= 48
            if not near_bottom:
                return
            QtCore.QTimer.singleShot(0, lambda: scrollbar.setValue(scrollbar.maximum()))

        def _current_theme(self) -> ShellThemeTokens:
            return self._display_theme

        def _record_perf_sample(self, name: str, duration_ms: float) -> None:
            sample_bucket = self._perf_samples.get(name)
            if sample_bucket is None:
                return
            sample_bucket.append(max(0.0, float(duration_ms)))

        def _record_event_burst(self, item_count: int) -> None:
            normalized = max(0, int(item_count))
            self._perf_counters["last_event_burst_items"] = normalized
            self._perf_counters["max_event_burst_items"] = max(
                self._perf_counters["max_event_burst_items"],
                normalized,
            )

        def _record_intro_animation_batch(self, item_count: int) -> None:
            normalized = max(0, int(item_count))
            self._perf_counters["last_intro_animation_batch"] = normalized
            self._perf_counters["max_intro_animation_batch"] = max(
                self._perf_counters["max_intro_animation_batch"],
                normalized,
            )

        def _performance_lines(self) -> list[str]:
            lines = ["shell_perf:"]
            for key in (
                "orb_paint_ms",
                "backdrop_paint_ms",
                "shell_apply_ms",
                "dashboard_apply_ms",
                "resize_ms",
                "dock_sync_ms",
            ):
                samples = list(self._perf_samples[key])
                if samples:
                    lines.append(f"{key}: avg={_mean(samples):.2f} max={max(samples):.2f} samples={len(samples)}")
                else:
                    lines.append(f"{key}: avg=0.00 max=0.00 samples=0")
            lines.append(
                "event_burst_items: "
                f"last={self._perf_counters['last_event_burst_items']} "
                f"max={self._perf_counters['max_event_burst_items']}"
            )
            lines.append(
                "intro_animation_batch: "
                f"last={self._perf_counters['last_intro_animation_batch']} "
                f"max={self._perf_counters['max_intro_animation_batch']}"
            )
            return lines

        def _refresh_debug_surface(self) -> None:
            self._debug.setPlainText(
                "\n".join(
                    (
                        f"last_stage: {self._app_state.last_stage}",
                        f"event_count: {self._app_state.event_count}",
                        f"dropped_events: {self._app_state.dropped_events}",
                        f"last_notice: {self._app_state.last_notice}",
                        f"summary: {self._shell_state.current_task_summary}",
                        *self._coding_debug_lines(),
                        *self._performance_lines(),
                    )
                )
            )

        def _queue_theme_transition(self) -> None:
            self._target_theme = _theme(self._shell_state, self._ui_preferences)
            if bool(self._ui_preferences.get("reduced_motion", False)):
                self._display_theme = self._target_theme

        def _advance_theme_transition(self) -> bool:
            if self._display_theme == self._target_theme:
                return False
            factor = (
                1.0
                if bool(self._ui_preferences.get("reduced_motion", False))
                else 0.30
                if bool(self._ui_preferences.get("reduced_effects_mode", False))
                or bool(self._ui_preferences.get("low_resource_mode", False))
                else 0.16
            )
            next_theme = _blend_theme(self._display_theme, self._target_theme, factor)
            if next_theme == self._display_theme:
                next_theme = self._target_theme
            if next_theme == self._display_theme:
                return False
            self._display_theme = next_theme
            return True

        @QtCore.Slot(float)
        def _on_animation_tick(self, phase: float) -> None:
            del phase
            if self._advance_theme_transition():
                self._apply_theme()
            if self._notification_flash > 0.0:
                decay = 0.05 if not bool(self._ui_preferences.get("reduced_motion", False)) else 0.12
                self._notification_flash = max(0.0, self._notification_flash - decay)
            pulse = 0.5 + (math.sin(self._animation_clock.phase * 1.35) * 0.5)
            notification_opacity = 0.7 + (0.3 * max(self._notification_flash, pulse * self._notification_flash))
            self._notification_effect.setOpacity(
                1.0 if bool(self._ui_preferences.get("reduced_motion", False)) else notification_opacity
            )
            self._refresh_live_badge_styles()
            if self.centralWidget() is not None:
                self.centralWidget().update()

        def _refresh_live_badge_styles(self) -> None:
            if not self._activity_chip_widgets:
                return
            theme = self._current_theme()
            pulse = 0.5 + (math.sin(self._animation_clock.phase * 1.6) * 0.5)
            reduced_motion = bool(self._ui_preferences.get("reduced_motion", False))
            for label, active, tone in self._activity_chip_widgets:
                color = self._tone_color(tone)
                border = color if active else theme["edge"]
                if active and not reduced_motion:
                    alpha = 0.10 + (0.12 * pulse)
                else:
                    alpha = 0.05
                label.setStyleSheet(
                    "QLabel {"
                    f"color: {color}; background: rgba(255,255,255,{alpha:.2f});"
                    f"border: 1px solid {border}; border-radius: 13px; padding: 7px 14px;"
                    "font-size: 12px; font-weight: 700; letter-spacing: 0.3px;"
                    "}"
                )

        def _refresh_readiness_surfaces(self) -> None:
            self._show_bottom_tab(3)
            if self._request_action is not None:
                self._request_action("readiness.refresh", {})
                self._request_action("capabilities.refresh", {})
                self._request_action("models.refresh", {})

        def _export_support_bundle(self) -> None:
            if self._request_action is None:
                return
            self._request_action(
                "support.export_bundle",
                {"path": self._default_export_path("support_bundle", "")},
            )

        def _export_selected_trace(self) -> None:
            if self._request_action is None:
                return
            task_id = str(self._app_state.selected_task.task_id or "").strip()
            if not task_id and self._history_list.currentItem() is not None:
                task_id = str(
                    self._history_list.currentItem().data(QtCore.Qt.ItemDataRole.UserRole) or ""
                ).strip()
            if not task_id:
                task_id = str(self._app_state.active_task.task_id or "").strip()
            if not task_id:
                return
            self._show_bottom_tab(0)
            self._request_action(
                "history.export_task_debug",
                {
                    "task_id": task_id,
                    "path": self._default_export_path(task_id.replace(":", "_"), ".md"),
                },
            )

        def _persist_settings_update(self, **sections: dict[str, Any]) -> None:
            profile = self._app_state.user_settings
            updated_sections: dict[str, dict[str, Any]] = {}
            for section_name, values in sections.items():
                current_values = dict(getattr(profile, section_name))
                current_values.update(values)
                updated_sections[section_name] = current_values
            next_profile = replace(profile, **updated_sections)
            if self._save_settings is not None:
                self._save_settings(next_profile)
            self.apply_dashboard_state(replace(self._app_state, user_settings=next_profile))

        def _set_reasoning_preset(self, mode: str, minutes: int) -> None:
            bounded_minutes = max(1, min(720, int(minutes)))
            self._slider.setValue(bounded_minutes)
            if mode == "fast":
                self._persist_settings_update(
                    reasoning={"mode": "fast", "thinking_minutes": bounded_minutes},
                    long_horizon={"enabled": False},
                )
            elif mode == "deep":
                self._persist_settings_update(
                    reasoning={"mode": "deep", "thinking_minutes": bounded_minutes},
                    long_horizon={"enabled": False},
                )
            else:
                checkpoint = max(30, min(60, bounded_minutes))
                self._persist_settings_update(
                    reasoning={"mode": "deep", "thinking_minutes": bounded_minutes},
                    long_horizon={
                        "enabled": True,
                        "wall_clock_minutes": bounded_minutes,
                        "cycle_budget_minutes": checkpoint,
                        "checkpoint_interval_minutes": checkpoint,
                    },
                )

        def _sync_slider_preference(self) -> None:
            minutes = int(self._slider.value())
            current_mode = str(self._app_state.user_settings.reasoning.get("mode", "auto")).strip() or "auto"
            if minutes > 120:
                self._set_reasoning_preset("long_horizon", minutes)
                return
            next_mode = "fast" if minutes <= 15 else "deep" if current_mode in {"deep", "fast"} else current_mode
            self._persist_settings_update(
                reasoning={"mode": next_mode, "thinking_minutes": minutes},
                long_horizon={"enabled": False},
            )

        def _toggle_local_only(self, enabled: bool) -> None:
            updates: dict[str, dict[str, Any]] = {
                "coding": {"local_only": bool(enabled)},
            }
            if enabled:
                updates["runtime"] = {"allow_web_fallback": False}
                updates["retrieval"] = {"allow_web_fallback": False}
                updates["cloud"] = {"mode": "disabled"}
            self._persist_settings_update(**updates)

        def _toggle_web_access(self, enabled: bool) -> None:
            allow_web = bool(enabled) and not bool(self._local_only_toggle.isChecked())
            self._persist_settings_update(
                runtime={"allow_web_fallback": allow_web},
                retrieval={"allow_web_fallback": allow_web},
            )

        def _toggle_verification_priority(self, enabled: bool) -> None:
            self._persist_settings_update(
                reasoning={
                    "mode": "deep" if enabled else "auto",
                    "thinking_minutes": max(
                        int(self._app_state.user_settings.reasoning.get("thinking_minutes", 30) or 30),
                        30 if enabled else 5,
                    ),
                }
            )

        def _toggle_capability_session(self, enabled: bool) -> None:
            self._persist_settings_update(desktop={"enabled": bool(enabled)})

        def _toggle_cloud_helper(self, enabled: bool) -> None:
            self._persist_settings_update(
                cloud={"mode": "auxiliary_only" if enabled else "disabled"}
            )

        @staticmethod
        def _set_checkable_button_state(button: QtWidgets.QPushButton, checked: bool) -> None:
            button.blockSignals(True)
            button.setChecked(bool(checked))
            button.blockSignals(False)

        def _coding_roles(self) -> list[str]:
            roles = {
                str(role).strip()
                for role in dict(self._app_state.user_settings.coding.get("preferred_models_by_role", {}))
                if str(role).strip()
            }
            for decision in self._app_state.model_registry_view.last_route_decisions:
                coding_role = str(decision.metadata.get("coding_role", "")).strip()
                if coding_role:
                    roles.add(coding_role)
            for registration in self._app_state.model_registry_view.registrations:
                if str(getattr(registration.role, "value", registration.role)) != "code_specialist":
                    continue
                roles.update(
                    str(item).strip()
                    for item in registration.metadata.get("coding_roles", ())
                    if str(item).strip()
                )
            for route_entry in self._shell_state.active_route_summary:
                role_name = str(route_entry).split(":", 1)[0].strip()
                if role_name:
                    roles.add(role_name)
            return sorted(roles)

        def _coding_registrations_for_role(self, role_name: str) -> list[Any]:
            registrations = [
                registration
                for registration in self._app_state.model_registry_view.registrations
                if str(getattr(registration.role, "value", registration.role)) == "code_specialist"
            ]
            matching = [
                registration
                for registration in registrations
                if role_name in {str(item) for item in registration.metadata.get("coding_roles", ())}
                or f"code:{role_name}" in set(registration.supported_capabilities)
            ]
            return matching or registrations

        def _route_decision_for_role(self, role_name: str) -> Any | None:
            return next(
                (
                    decision
                    for decision in self._app_state.model_registry_view.last_route_decisions
                    if str(decision.metadata.get("coding_role", "")).strip() == role_name
                ),
                None,
            )

        def _on_coding_role_changed(self, _index: int) -> None:
            if self._suppress_coding_route_signals:
                return
            self._selected_coding_role = self._coding_role_combo.currentData() or self._coding_role_combo.currentText()
            self._refresh_coding_route_controls()

        def _on_coding_model_changed(self, _index: int) -> None:
            if self._suppress_coding_route_signals:
                return
            role_name = str(
                self._coding_role_combo.currentData() or self._coding_role_combo.currentText()
            ).strip()
            registration_id = str(self._coding_model_combo.currentData() or "").strip()
            if not role_name or not registration_id:
                return
            preferred = dict(self._app_state.user_settings.coding.get("preferred_models_by_role", {}))
            preferred[role_name] = registration_id
            self._persist_settings_update(coding={"preferred_models_by_role": preferred})

        def _copy_to_clipboard(self, text: str) -> None:
            clipboard = QtWidgets.QApplication.clipboard()
            if clipboard is not None:
                clipboard.setText(text)

        def _copy_final_answer(self) -> None:
            self._copy_to_clipboard(self._final_answer_body.text().strip())

        def _copy_final_references(self) -> None:
            self._copy_to_clipboard(self._final_answer_refs.text().strip())

        @staticmethod
        def _format_stage_label(stage: str) -> str:
            return str(stage or "idle").replace(".", " / ").replace("_", " ").strip().title()

        @staticmethod
        def _timeline_category(stage: str) -> str:
            normalized = str(stage or "").strip()
            if not normalized:
                return "General"
            if "web_lookup" in normalized or "fallback" in normalized:
                return "Fallback"
            if normalized.startswith("researcher.") or "researcher" in normalized:
                return "Retrieval"
            if normalized.startswith("pipeline.reasoner"):
                return "Reasoning"
            if normalized.startswith("pipeline.critic"):
                return "Verification"
            if normalized.startswith("pipeline.compressor"):
                return "Compression"
            if normalized.startswith("pipeline.long_horizon"):
                return "Checkpoint"
            if normalized.startswith("coding."):
                return "Coding"
            if "degraded" in normalized:
                return "Degraded"
            if "approval" in normalized or "session" in normalized:
                return "Control"
            if normalized.startswith("pipeline.completed"):
                return "Response"
            return "General"

        @staticmethod
        def _format_elapsed(seconds: float) -> str:
            total_seconds = max(0, int(seconds or 0.0))
            hours, remainder = divmod(total_seconds, 3600)
            minutes, seconds_only = divmod(remainder, 60)
            if hours:
                return f"{hours}h {minutes}m"
            if minutes:
                return f"{minutes}m {seconds_only}s"
            return f"{seconds_only}s"

        @staticmethod
        def _route_primary_label(route_summary: tuple[str, ...]) -> str:
            return str(route_summary[0]).strip() if route_summary else ""

        @staticmethod
        def _workspace_chrome_label(workspace_mode: str) -> str:
            return "CODING STATION" if workspace_mode == "coding_workspace" else "ASSISTANT STATION"

        def _refresh_window_chrome(self) -> None:
            shell_state = self._shell_state
            app_state = self._app_state
            profile_name = str(app_state.user_settings.profile_name or "default").strip() or "default"
            route_label = self._route_primary_label(shell_state.active_route_summary)
            if not route_label and shell_state.active_agent:
                route_label = f"lead:{shell_state.active_agent}"
            state_label = str(shell_state.status_text or shell_state.orb_mode or "Ready").strip() or "Ready"
            signal_label = "Local Ready"
            if shell_state.degraded_reason or shell_state.fallback_reason:
                signal_label = "Fallback Active"
            elif shell_state.approval_pending:
                signal_label = "Approval Hold"
            elif shell_state.verifier_state in {"running", "verifying", "verified"}:
                signal_label = "Verification"
            elif shell_state.long_horizon_state not in {"", "idle", "inactive"}:
                signal_label = "Long Horizon"
            elif shell_state.cloud_helper_state not in {"", "disabled", "inactive"}:
                signal_label = "Cloud Helper"
            elif shell_state.capability_session_state not in {"", "inactive"}:
                signal_label = "Capability Session"

            self._chrome_workspace_badge.setText(self._workspace_chrome_label(shell_state.workspace_mode))
            self._chrome_profile_badge.setText(f"PROFILE {profile_name.upper()}")
            self._chrome_route_badge.setText(f"ROUTE {route_label.upper()}" if route_label else "ROUTE STANDBY")
            self._chrome_status_badge.setText(f"STATE {state_label.upper()}")
            self._chrome_signal_badge.setText(signal_label.upper())
            self._chrome_profile_badge.setVisible(bool(profile_name))
            self._chrome_route_badge.setVisible(True)
            self._chrome_status_badge.setVisible(True)
            self._chrome_signal_badge.setVisible(True)

        def _tone_color(self, tone: str) -> str:
            theme = self._current_theme()
            return {
                "warning": theme["warning"],
                "danger": theme["danger"],
                "accent": theme["highlight"],
                "success": theme["success"],
                "muted": theme["muted"],
                "info": theme["accent"],
            }.get(tone, theme["text"])

        def _badge(self, text: str, tone: str = "info", parent: QtWidgets.QWidget | None = None) -> QtWidgets.QLabel:
            label = QtWidgets.QLabel(text, parent)
            color = self._tone_color(tone)
            theme = self._current_theme()
            label.setStyleSheet(
                "QLabel {"
                f"color: {color}; background: rgba(255,255,255,0.06);"
                f"border: 1px solid {theme['edge']}; border-radius: 13px; padding: 7px 12px;"
                "font-size: 12px; font-weight: 600; letter-spacing: 0.3px;"
                "}"
            )
            return label

        def _set_badges(
            self,
            layout: QtWidgets.QBoxLayout,
            parent: QtWidgets.QWidget,
            badges: list[tuple[str, str, str]],
            *,
            centered: bool = False,
        ) -> None:
            _clear(layout)
            if centered:
                layout.addStretch(1)
            for text, tone, tooltip in badges:
                if not str(text).strip():
                    continue
                label = self._badge(text, tone=tone, parent=parent)
                if tooltip:
                    label.setToolTip(tooltip)
                layout.addWidget(label)
            layout.addStretch(1)

        def _populate_list(
            self,
            widget: QtWidgets.QListWidget,
            entries: list[tuple[str, str, str]],
            *,
            user_data: list[str] | None = None,
            details: list[str] | None = None,
        ) -> None:
            current_value = ""
            current_item = widget.currentItem()
            if current_item is not None:
                current_value = str(current_item.data(QtCore.Qt.ItemDataRole.UserRole) or "")
            widget.clear()
            theme = self._current_theme()
            selected_row = -1
            for index, (headline, detail, tone) in enumerate(entries):
                text = f"{headline}\n{detail}".strip() if detail else headline
                item = QtWidgets.QListWidgetItem(text)
                item.setToolTip(detail or headline)
                item.setForeground(QtGui.QColor(self._tone_color(tone)))
                if user_data is not None and index < len(user_data):
                    value = user_data[index]
                    item.setData(QtCore.Qt.ItemDataRole.UserRole, value)
                    if value and value == current_value:
                        selected_row = index
                if details is not None and index < len(details):
                    item.setData(_ITEM_DETAIL_ROLE, details[index])
                widget.addItem(item)
            if widget.count() and selected_row == -1 and current_value:
                for index in range(widget.count()):
                    if str(widget.item(index).data(QtCore.Qt.ItemDataRole.UserRole) or "") == current_value:
                        selected_row = index
                        break
            if widget.count():
                widget.setCurrentRow(selected_row if selected_row >= 0 else 0)

        def _approval_risk_summary(self) -> tuple[str, str]:
            session = self._app_state.local_task_session
            summary = " ".join(
                filter(
                    None,
                    (
                        self._shell_state.approval_prompt_summary,
                        session.last_action_summary,
                        session.current_target,
                    ),
                )
            ).lower()
            if (
                self._shell_state.observation_tier in {"vision_on_step", "continuous_capture"}
                or any(token in summary for token in ("delete", "move", "write", "type", "input", "kill", "shell"))
            ):
                return ("Elevated Risk", "Desktop or file-affecting work is paused until you explicitly approve it.")
            if session.current_target or session.control_mode != "local_task":
                return ("Guarded Risk", "The action is scoped to a live target and requires operator confirmation.")
            return ("Review Required", "The current capability policy requires confirmation before the next step.")

        @staticmethod
        def _bool_word(value: bool) -> str:
            return "yes" if value else "no"

        @staticmethod
        def _looks_active_state(value: str) -> bool:
            normalized = str(value or "").strip().lower()
            if not normalized:
                return False
            if normalized in {
                "offline",
                "idle",
                "ready",
                "inactive",
                "completed",
                "complete",
                "succeeded",
                "failed",
                "stopped",
                "cancelled",
                "canceled",
            }:
                return False
            return not normalized.endswith(
                (".completed", ".complete", ".succeeded", ".failed", ".cancelled", ".canceled", ".stopped")
            )

        def _recent_coding_results(self) -> list[Any]:
            app_state = self._app_state
            results = list(app_state.recent_coding_results)
            current = app_state.coding_output
            if current.request_id and current.status not in {"", "idle"}:
                existing_ids = {str(result.request_id) for result in results}
                if current.request_id not in existing_ids:
                    results.insert(0, current)
            return results[:8]

        def _recent_coding_practice_sessions(self) -> list[Any]:
            app_state = self._app_state
            sessions = list(app_state.recent_coding_practice_sessions)
            current = app_state.coding_practice
            if current.session_id and current.status not in {"", "idle"}:
                existing_ids = {str(session.session_id) for session in sessions}
                if current.session_id not in existing_ids:
                    sessions.insert(0, current)
            return sessions[:6]

        @staticmethod
        def _score_trend_text(scores: list[float]) -> str:
            if not scores:
                return "(no history)"
            if len(scores) == 1:
                return f"{scores[0]:.2f}"
            delta = scores[0] - scores[-1]
            direction = "up" if delta > 0.02 else "down" if delta < -0.02 else "flat"
            return f"{scores[0]:.2f} -> {scores[-1]:.2f} ({direction} {abs(delta):.2f})"

        @staticmethod
        def _model_performance_lines(results: list[Any]) -> list[str]:
            grouped: dict[tuple[str, str], list[float]] = {}
            for result in results:
                task_type = str(getattr(getattr(result, "task_type", ""), "value", getattr(result, "task_type", "")))
                role_assignments = dict(getattr(result, "role_assignments", {}) or {})
                for role_name, model_name in role_assignments.items():
                    key = (task_type or "unknown", f"{role_name}={model_name}")
                    grouped.setdefault(key, []).append(float(result.quality_report.quality_score))
            lines = [
                f"{task_type}: {model_role} avg {_mean(scores):.2f} over {len(scores)} run(s)"
                for (task_type, model_role), scores in sorted(
                    grouped.items(),
                    key=lambda item: (_mean(item[1]), len(item[1])),
                    reverse=True,
                )[:4]
            ]
            return lines or ["No recent per-model coding performance recorded."]

        @staticmethod
        def _primary_coding_artifacts(artifacts: tuple[Any, ...]) -> list[Any]:
            debug_types = {"sandbox_trace", "stdout", "stderr"}
            return [artifact for artifact in artifacts if str(getattr(artifact, "artifact_type", "")).strip() not in debug_types]

        @staticmethod
        def _coding_debug_artifacts(artifacts: tuple[Any, ...]) -> list[Any]:
            debug_types = {"sandbox_trace", "stdout", "stderr"}
            return [artifact for artifact in artifacts if str(getattr(artifact, "artifact_type", "")).strip() in debug_types]

        def _refresh_hero_surfaces(self) -> None:
            shell_state = self._shell_state
            agent_bits: list[str] = []
            if shell_state.active_agent:
                agent_bits.append(
                    "Lead "
                    + str(shell_state.active_agent).replace("_", " ").replace(".", " ").title()
                )
            if shell_state.secondary_agents:
                agent_bits.append(
                    "Support "
                    + ", ".join(
                        str(agent).replace("_", " ").replace(".", " ").title()
                        for agent in shell_state.secondary_agents[:4]
                    )
                )
            if shell_state.active_tools:
                agent_bits.append(
                    "Tools " + ", ".join(str(tool).replace("_", " ") for tool in shell_state.active_tools[:4])
                )
            hero_agents = "   |   ".join(agent_bits)
            self._hero_agents.setText(hero_agents)
            self._hero_agents.setVisible(bool(hero_agents))

        def _refresh_long_horizon_tray(self) -> None:
            shell_state = self._shell_state
            task = self._app_state.active_task
            configured = self._app_state.user_settings.long_horizon
            show_tray = bool(
                shell_state.panel_visibility_state.get("long_horizon_tray", False)
                and (
                    task.long_horizon_session_id
                    or task.execution_mode == "long_horizon"
                    or self._looks_active_state(shell_state.long_horizon_state)
                    or task.long_horizon_completed_cycles
                )
            )
            phase = (
                task.long_horizon_current_phase
                or shell_state.long_horizon_state
                or task.long_horizon_status
                or "planned"
            )
            cycle_budget = int(
                task.long_horizon_cycle_budget_minutes
                or configured.get("cycle_budget_minutes", 120)
                or 120
            )
            checkpoint_minutes = int(
                task.long_horizon_checkpoint_interval_minutes
                or configured.get("checkpoint_interval_minutes", cycle_budget)
                or cycle_budget
            )
            summary = (
                f"Session {task.long_horizon_session_id or '(planned)'} | "
                f"cycle budget {cycle_budget} min | checkpoint every {checkpoint_minutes} min"
            )
            metrics = [
                f"Elapsed {self._format_elapsed(task.long_horizon_elapsed_seconds)}",
                f"Checkpoints {task.long_horizon_completed_cycles}/{task.long_horizon_total_cycles or '?'}",
                (
                    "Candidates "
                    f"{task.long_horizon_initial_candidate_count}->{task.long_horizon_peak_candidate_count}"
                    f" (+{task.long_horizon_additional_candidate_count})"
                ),
                (
                    "Evidence "
                    f"{task.long_horizon_initial_supporting_evidence_count}"
                    f" (+{task.long_horizon_additional_supporting_evidence_count})"
                ),
                f"Verification passes {task.long_horizon_total_verification_passes}",
                f"Repairs {task.long_horizon_total_repairs}",
                f"Duty cycle {task.long_horizon_duty_cycle_ratio:.2f}",
                (
                    "Advisory "
                    f"{task.long_horizon_advisory_accepted_count}/"
                    f"{task.long_horizon_advisory_requested_count} accepted"
                ),
            ]
            delta_bits = []
            if task.long_horizon_confidence_gain:
                delta_bits.append(f"Confidence gain {task.long_horizon_confidence_gain:+.2f}")
            if task.long_horizon_validity_improved:
                delta_bits.append("Validity improved")
            if task.long_horizon_additional_candidate_count:
                delta_bits.append(
                    f"Extra time bought {task.long_horizon_additional_candidate_count} more candidates"
                )
            if task.long_horizon_additional_supporting_evidence_count:
                delta_bits.append(
                    f"and {task.long_horizon_additional_supporting_evidence_count} more evidence refreshes"
                )
            if task.long_horizon_early_stop_reason:
                delta_bits.append(f"Early stop: {task.long_horizon_early_stop_reason}")
            if task.long_horizon_advisory_entries:
                delta_bits.append("Advisories: " + " | ".join(task.long_horizon_advisory_entries[:3]))
            self._long_horizon_tray.setVisible(show_tray)
            self._long_horizon_title.setText("Long-Horizon Tray")
            self._long_horizon_phase.setText(self._format_stage_label(phase))
            self._long_horizon_summary.setText(summary)
            self._long_horizon_metrics.setText("   |   ".join(metrics))
            self._long_horizon_delta.setText(
                "   |   ".join(delta_bits) if delta_bits else "Extra time has not produced new deltas yet."
            )

        def _refresh_dock_controls(self) -> None:
            shell_state = self._shell_state
            app_state = self._app_state
            profile = app_state.user_settings
            session = app_state.local_task_session
            task = app_state.active_task
            reasoning_mode = str(profile.reasoning.get("mode", "auto")).strip() or "auto"
            long_enabled = bool(profile.long_horizon.get("enabled", False) or task.long_horizon_session_id)
            web_enabled = bool(
                profile.runtime.get("allow_web_fallback", True)
                or profile.retrieval.get("allow_web_fallback", True)
            )
            local_only = bool(profile.coding.get("local_only", False)) or (
                not web_enabled and str(profile.cloud.get("mode", "disabled")) == "disabled"
            )
            self._set_checkable_button_state(self._fast, reasoning_mode == "fast" and not long_enabled)
            self._set_checkable_button_state(self._deep, reasoning_mode == "deep" and not long_enabled)
            self._set_checkable_button_state(self._long, long_enabled)
            self._set_checkable_button_state(self._local_only_toggle, local_only)
            self._set_checkable_button_state(self._web_toggle, web_enabled and not local_only)
            self._set_checkable_button_state(self._verify_toggle, reasoning_mode == "deep" or long_enabled)
            self._set_checkable_button_state(
                self._capability_toggle,
                bool(profile.desktop.get("enabled", False) or session.session_id),
            )
            self._set_checkable_button_state(
                self._cloud_toggle,
                str(profile.cloud.get("mode", "disabled")) != "disabled",
            )
            has_session = bool(session.session_id)
            has_long_horizon = bool(task.long_horizon_session_id)
            self._pause.setEnabled((has_session and session.status == "running") or has_long_horizon)
            self._resume.setEnabled((has_session and session.status == "paused") or has_long_horizon)
            self._stop.setEnabled(has_session or has_long_horizon)
            self._web_toggle.setEnabled(not local_only)
            self._cloud_toggle.setEnabled(not local_only)
            mode_label = "Long Horizon" if long_enabled else reasoning_mode.title()
            layout_label = (
                "Adaptive"
                if bool(self._ui_preferences.get("adaptive_drawers", True))
                else "Pinned"
            )
            dock_summary = [
                f"Mode {mode_label}",
                f"Drawers {layout_label}",
                f"Web {'on' if web_enabled and not local_only else 'off'}",
                f"Capability {session.status or ('armed' if profile.desktop.get('enabled', False) else 'off')}",
                f"Cloud {shell_state.cloud_helper_state}",
            ]
            if shell_state.degraded_reason:
                dock_summary.append(f"Degraded {shell_state.degraded_reason}")
            self._dock_status.setText("   |   ".join(dock_summary))

        def _refresh_coding_route_controls(self) -> None:
            registry = self._app_state.model_registry_view
            role_names = self._coding_roles()
            current_role = self._selected_coding_role if self._selected_coding_role in role_names else ""
            if not current_role and role_names:
                current_role = role_names[0]
            self._selected_coding_role = current_role
            self._suppress_coding_route_signals = True
            try:
                self._coding_role_combo.clear()
                for role_name in role_names:
                    self._coding_role_combo.addItem(role_name.replace("_", " ").title(), role_name)
                if current_role:
                    index = self._coding_role_combo.findData(current_role)
                    if index >= 0:
                        self._coding_role_combo.setCurrentIndex(index)

                self._coding_model_combo.clear()
                registrations = self._coding_registrations_for_role(current_role) if current_role else []
                decision = self._route_decision_for_role(current_role) if current_role else None
                preferred = str(
                    self._app_state.user_settings.coding.get("preferred_models_by_role", {}).get(current_role, "")
                ).strip()
                preferred = preferred or str(
                    getattr(decision, "selected_registration_id", "")
                    or (
                        f"{getattr(decision, 'selected_backend', '')}:"
                        f"{getattr(decision, 'selected_model_identifier', '')}"
                    )
                ).strip(":")
                for registration in registrations:
                    identifier = str(registration.registration_id or "")
                    display_value = f"{registration.backend}:{registration.model_identifier}"
                    label = (
                        f"{identifier or display_value} | "
                        f"{display_value}"
                    )
                    self._coding_model_combo.addItem(label, identifier or display_value)
                if preferred:
                    index = self._coding_model_combo.findData(preferred)
                    if index == -1:
                        index = self._coding_model_combo.findText(preferred)
                    if index >= 0:
                        self._coding_model_combo.setCurrentIndex(index)
            finally:
                self._suppress_coding_route_signals = False
            self._coding_role_combo.setEnabled(bool(role_names))
            self._coding_model_combo.setEnabled(bool(registrations))

            warm_keys = {
                str(key)
                for cache in registry.cache_snapshots
                for key in cache.warm_keys
                if str(key).strip()
            }
            decision = self._route_decision_for_role(current_role) if current_role else None
            selected_registration = str(
                getattr(decision, "selected_registration_id", "")
                or (self._coding_model_combo.currentData() or "")
            ).strip()
            is_warm = bool(selected_registration and selected_registration in warm_keys)
            route_lines = [
                (
                    f"{str(route_decision.metadata.get('coding_role', route_decision.requested_role.value)).replace('_', ' ')}",
                    (
                        f"{route_decision.selected_registration_id or '(none)'} | "
                        f"{route_decision.selected_model_identifier or route_decision.fallback_reason or '(unrouted)'}"
                    ),
                    "warning" if route_decision.used_fallback else "info",
                )
                for route_decision in registry.last_route_decisions
                if str(route_decision.metadata.get("coding_role", "")).strip()
                or str(route_decision.requested_role.value) == "code_specialist"
            ] or [("Coding routes", "No coding routes are active.", "muted")]
            self._populate_list(self._coding_routes_list, route_lines)
            self._coding_route_summary.setText(
                "   |   ".join(
                    filter(
                        None,
                        (
                            f"Role {current_role.replace('_', ' ')}" if current_role else "",
                            f"Warm {'yes' if is_warm else 'no'}" if current_role else "",
                            (
                                f"Heavy slots {len(registry.active_heavy_roles)}/{registry.heavy_slot_limit}"
                                if registry.heavy_slot_limit
                                else ""
                            ),
                        ),
                    )
                )
                or "No coding route selection is active."
            )
            fallback_reason = str(getattr(decision, "fallback_reason", "") or "").strip()
            self._coding_route_note.setText(
                "   |   ".join(
                    filter(
                        None,
                        (
                            f"Selected {selected_registration}" if selected_registration else "",
                            f"Fallback {fallback_reason}" if fallback_reason else "",
                            f"Governor {registry.governor_summary}" if registry.governor_summary else "",
                        ),
                    )
                )
                or "Select a coding role to inspect or pin its preferred local model."
            )

        def _refresh_context_surfaces(self) -> None:
            shell_state = self._shell_state
            app_state = self._app_state
            session = app_state.local_task_session
            cloud_mode = str(app_state.user_settings.cloud.get("mode", "auxiliary_only")).replace("_", " ")
            desktop_policy = str(app_state.user_settings.desktop.get("approval_policy", "approve_risky_only")).replace("_", " ")
            policy_bits = [
                (
                    f"Session {session.status or 'inactive'}",
                    "accent" if session.session_id else "muted",
                    session.label or "No active local task session.",
                ),
                (
                    f"Target {session.current_target}",
                    "info",
                    session.last_action_summary or session.current_target,
                )
                if session.current_target
                else ("", "info", ""),
                (
                    f"Approval {len(session.pending_approval_summaries)}",
                    "warning",
                    shell_state.approval_prompt_summary or "No approval pending.",
                )
                if session.pending_approval_summaries
                else ("", "warning", ""),
                (
                    f"Observation {shell_state.observation_tier}",
                    "accent" if shell_state.observation_tier != "screenshot_on_demand" else "muted",
                    session.last_observation_summary or shell_state.observation_tier,
                ),
                (
                    f"Policy {desktop_policy}",
                    "info",
                    f"Web fallback {self._bool_word(bool(app_state.user_settings.runtime.get('allow_web_fallback', True)))}"
                    f" | cloud {cloud_mode}",
                ),
                (
                    f"Last action {session.last_action_summary}",
                    "info",
                    session.last_control_reason or session.last_action_summary,
                )
                if session.last_action_summary
                else ("", "info", ""),
            ]
            self._set_badges(
                self._policy_context_layout,
                self._policy_context_bar,
                [entry for entry in policy_bits if entry[0]],
                centered=True,
            )
            show_context = any(entry[0] for entry in policy_bits)
            self._policy_context_bar.setVisible(show_context)

            risk_label, risk_reason = self._approval_risk_summary()
            show_approval = bool(shell_state.approval_pending)
            summary = (
                shell_state.approval_prompt_summary
                or session.last_action_summary
                or "A bounded capability action is waiting for explicit approval."
            )
            target = session.current_target or session.last_observation_summary or shell_state.current_file or "(unknown target)"
            reason = (
                f"Why blocked: policy {desktop_policy} is active."
                f" {risk_reason}"
            )
            self._approval_overlay.setVisible(show_approval)
            self._approval_title.setText("Approval Required")
            self._approval_risk.setText(risk_label)
            self._approval_summary.setText(f"Requested action: {summary}")
            self._approval_target.setText(f"Target: {target}")
            self._approval_reason.setText(reason)
            has_local_session = bool(session.session_id)
            self._approval_pause_button.setVisible(has_local_session)
            self._approval_resume_button.setVisible(has_local_session)
            self._approval_stop_button.setVisible(bool(session.session_id or app_state.active_task.long_horizon_session_id))
            self._approval_pause_button.setEnabled(has_local_session and session.status == "running")
            self._approval_resume_button.setEnabled(has_local_session and session.status == "paused")
            self._approval_stop_button.setEnabled(bool(session.session_id or app_state.active_task.long_horizon_session_id))

            is_coding = shell_state.workspace_mode == "coding_workspace"
            coding_badges = [
                (
                    f"File {shell_state.current_file}",
                    "accent",
                    shell_state.current_file,
                )
                if shell_state.current_file
                else ("", "accent", ""),
                (
                    f"Project {shell_state.current_project}",
                    "info",
                    shell_state.current_project,
                )
                if shell_state.current_project
                else ("", "info", ""),
                (
                    f"Model {self._route_primary_label(shell_state.active_route_summary)}",
                    "accent",
                    "Primary coding route",
                )
                if self._route_primary_label(shell_state.active_route_summary)
                else ("", "accent", ""),
                (
                    f"Sandbox {shell_state.sandbox_state}",
                    "warning" if shell_state.sandbox_state not in {"completed", "idle"} else "success",
                    "Current coding sandbox state",
                ),
            ]
            self._set_badges(
                self._coding_dock_layout,
                self._coding_dock_bar,
                [entry for entry in coding_badges if entry[0]],
                centered=False,
            )
            if is_coding:
                for label, minutes in (("Patch", 20), ("Review+", 45), ("Validate", 90)):
                    button = QtWidgets.QPushButton(label, self._coding_dock_bar)
                    button.clicked.connect(lambda _checked=False, value=minutes: self._slider.setValue(value))
                    self._coding_dock_layout.insertWidget(max(0, self._coding_dock_layout.count() - 1), button)
            self._coding_dock_bar.setVisible(is_coding)

        def _refresh_drawer_surfaces(self) -> None:
            shell_state = self._shell_state
            app_state = self._app_state
            task = app_state.active_task
            session = app_state.local_task_session
            registry = app_state.model_registry_view
            health = app_state.runtime_health
            coding_output = app_state.coding_output
            coding_practice = app_state.coding_practice
            recent_coding_results = self._recent_coding_results()
            recent_practice_sessions = self._recent_coding_practice_sessions()
            is_coding = shell_state.workspace_mode == "coding_workspace"

            self._left_dock.setWindowTitle("Coding Timeline" if is_coding else "Task Timeline")
            self._right_dock.setWindowTitle("Coding Insights" if is_coding else "Evidence And Insights")

            agent_entries = [
                (
                    component.title(),
                    f"{status.state.value} [{status.severity.value}] {status.message or '(no detail)'}",
                    "warning" if status.severity.value in {"medium", "high"} else "info",
                )
                for component, status in sorted(app_state.statuses.items())
            ] or [("Agents", "No live agent status yet.", "muted")]
            self._populate_list(self._agent_list, agent_entries)

            session_badges = [
                (
                    f"Approvals {len(session.pending_approval_summaries)}",
                    "warning" if session.pending_approval_summaries else "muted",
                    "Queued approval requests",
                ),
                (
                    f"Observation {session.effective_observation_tier or 'screenshot_on_demand'}",
                    "accent",
                    session.last_observation_summary or "Current observation tier",
                ),
                (
                    f"Control {session.control_mode or 'local_task'}",
                    "info",
                    session.last_control_reason or "Current control mode",
                ),
            ]
            self._set_badges(self._session_badges_layout, self._session_badges, session_badges, centered=False)
            self._session_summary.setText(
                "\n".join(
                    (
                        f"Session {session.session_id or '(none)'} is {session.status}.",
                        f"Target: {session.current_target or '(none)'}",
                        f"Last action: {session.last_action_summary or '(none)'}",
                        f"Long horizon: {task.long_horizon_session_id or '(none)'}",
                    )
                )
            )
            self._session_pause_button.setEnabled(bool(session.session_id) and session.status == "running")
            self._session_resume_button.setEnabled(bool(session.session_id) and session.status == "paused")
            self._session_stop_button.setEnabled(bool(session.session_id or task.long_horizon_session_id))
            self._session_kill_button.setEnabled(bool(session.session_id))
            session_entries = [
                ("Pending approvals", " | ".join(session.pending_approval_summaries) or "(none)", "warning"),
                ("Last control reason", session.last_control_reason or "(none)", "info"),
                ("Last observation", session.last_observation_summary or session.last_observation_status or "(none)", "info"),
                ("Continuous capture", f"active={self._bool_word(session.continuous_capture_active)} fps={session.continuous_capture_fps:.1f}", "muted"),
                ("Kill switch", self._bool_word(session.kill_switch_engaged), "danger" if session.kill_switch_engaged else "muted"),
                ("Last error", session.last_error or "(none)", "danger" if session.last_error else "muted"),
            ]
            if is_coding:
                session_entries.extend(
                    [
                        ("Coding state", shell_state.coding_state or "(idle)", "accent"),
                        ("Sandbox", shell_state.sandbox_state, "warning" if shell_state.sandbox_state not in {"idle", "completed", "ready"} else "info"),
                        ("Validation", shell_state.quality_gate_state, "warning" if shell_state.quality_gate_state not in {"idle", "passed"} else "success"),
                    ]
                )
            self._populate_list(self._session_diagnostics, session_entries)

            evidence_entries = [
                ("Local retrieval", str(task.local_result_count), "info"),
                ("Web retrieval", str(task.web_result_count), "warning" if task.web_result_count else "muted"),
                ("Supporting evidence", ", ".join(task.supporting_evidence_ids) if task.supporting_evidence_ids else "(none)", "accent"),
                ("Citations", ", ".join(task.citation_refs) if task.citation_refs else "(none)", "accent"),
                ("Web sources", ", ".join(task.web_source_refs) if task.web_source_refs else "(none)", "muted"),
            ]
            evidence_details = [
                "\n".join(
                    (
                        f"Question: {task.question or shell_state.current_task_summary or '(none)'}",
                        f"Local retrieval count: {task.local_result_count}",
                        f"Web retrieval count: {task.web_result_count}",
                        f"Supporting evidence: {', '.join(task.supporting_evidence_ids) if task.supporting_evidence_ids else '(none)'}",
                        f"Citations: {', '.join(task.citation_refs) if task.citation_refs else '(none)'}",
                        f"Web sources: {', '.join(task.web_source_refs) if task.web_source_refs else '(none)'}",
                    )
                )
                for _headline, _detail, _tone in evidence_entries
            ]
            self._populate_list(self._evidence_list, evidence_entries, details=evidence_details)
            self._evidence_panel.set_summary(
                f"{task.local_result_count} local | {task.web_result_count} web | {len(task.citation_refs)} citations"
            )
            self._evidence_panel.set_empty_detail("Select an evidence row to inspect retrieval and citation details.")

            provenance_entries = [
                ("Task id", task.task_id or "(none)", "info"),
                ("Stage", task.running_stage or app_state.last_stage or "(idle)", "accent"),
                ("Verifier", task.selected_verifier or "(none)", "accent"),
                ("Strategy", task.selected_strategy or "(none)", "muted"),
                ("Candidate score", f"{task.candidate_score:.2f}", "success" if task.candidate_score >= 0.75 else "warning"),
                ("Repairs", ", ".join(task.repair_actions) if task.repair_actions else "(none)", "info"),
                ("Failures", ", ".join(task.failure_categories) if task.failure_categories else "(none)", "danger" if task.failure_categories else "muted"),
            ]
            provenance_details = [
                "\n".join(
                    (
                        f"Task id: {task.task_id or '(none)'}",
                        f"Stage: {task.running_stage or app_state.last_stage or '(idle)'}",
                        f"Selected verifier: {task.selected_verifier or '(none)'}",
                        f"Selected strategy: {task.selected_strategy or '(none)'}",
                        f"Candidate score: {task.candidate_score:.2f}",
                        f"Repair actions: {', '.join(task.repair_actions) if task.repair_actions else '(none)'}",
                        f"Failure categories: {', '.join(task.failure_categories) if task.failure_categories else '(none)'}",
                    )
                )
                for _headline, _detail, _tone in provenance_entries
            ]
            self._populate_list(self._provenance_list, provenance_entries, details=provenance_details)
            self._provenance_panel.set_summary(
                f"Verifier {task.selected_verifier or '(none)'} | strategy {task.selected_strategy or '(none)'} | score {task.candidate_score:.2f}"
            )
            self._provenance_panel.set_empty_detail("Select a provenance row to inspect the current task lineage.")

            compression_entries = [
                (
                    insight.macro_name or insight.proposal_id,
                    f"gain {insight.estimated_gain:.2f} | pass {insight.validation_pass_rate:.2f} | {insight.validation_state}",
                    "success" if insight.accepted else "warning",
                )
                for insight in registry.compression_insights[:8]
            ] or [("Compression", shell_state.compression_state or "idle", "muted")]
            compression_details = [
                "\n".join(
                    (
                        f"Macro insight: {insight.macro_name or insight.proposal_id}",
                        f"Estimated gain: {insight.estimated_gain:.2f}",
                        f"Validation pass rate: {insight.validation_pass_rate:.2f}",
                        f"Validation state: {insight.validation_state}",
                        f"Accepted: {self._bool_word(insight.accepted)}",
                    )
                )
                for insight in registry.compression_insights[:8]
            ] or [f"Compression state: {shell_state.compression_state or 'idle'}"]
            self._populate_list(self._compression_list, compression_entries, details=compression_details)
            self._compression_panel.set_summary(
                f"{len(registry.compression_insights[:8])} visible proposals | state {shell_state.compression_state or 'idle'}"
            )
            self._compression_panel.set_empty_detail("Select a compressor row to inspect macro validation details.")

            optimizer_entries = [
                (
                    suggestion.summary or suggestion.suggestion_id,
                    f"{suggestion.kind.value} | {suggestion.disposition.value}",
                    "accent" if suggestion.disposition.value in {"requested", "accepted"} else "warning",
                )
                for suggestion in registry.recent_optimizer_suggestions[:8]
            ] or [
                (
                    "Optimizer",
                    ", ".join(task.advisor_summaries) if task.advisor_summaries else "No active optimizer advice.",
                    "muted",
                )
            ]
            optimizer_details = [
                "\n".join(
                    (
                        f"Suggestion: {suggestion.summary or suggestion.suggestion_id}",
                        f"Kind: {suggestion.kind.value}",
                        f"Confidence: {suggestion.confidence:.2f}",
                        f"Rationale: {suggestion.rationale or '(none)'}",
                        f"Targets: {', '.join(suggestion.target_components) if suggestion.target_components else '(none)'}",
                        f"Sources: {', '.join(suggestion.source_task_ids) if suggestion.source_task_ids else '(none)'}",
                    )
                )
                for suggestion in registry.recent_optimizer_suggestions[:8]
            ] or [
                "\n".join(
                    (
                        "Optimizer advisory is idle.",
                        f"Advisor summaries: {', '.join(task.advisor_summaries) if task.advisor_summaries else '(none)'}",
                    )
                )
            ]
            self._populate_list(self._optimizer_list, optimizer_entries, details=optimizer_details)
            self._optimizer_panel.set_summary(
                f"{len(registry.recent_optimizer_suggestions[:8])} live suggestions | state {shell_state.optimizer_state or 'idle'}"
            )
            self._optimizer_panel.set_empty_detail("Select an optimizer row to inspect rationale and routing impact.")

            control_entries = [
                ("Installed roles", str(len(registry.registrations)), "info"),
                ("Active heavy roles", ", ".join(registry.active_heavy_roles) if registry.active_heavy_roles else "(none)", "accent"),
                ("Governor", registry.governor_summary or "(inactive)", "warning" if registry.governor_active else "muted"),
            ]
            control_entries.extend(
                (
                    f"Route {decision.requested_role.value}",
                    (
                        f"{decision.selected_registration_id or '(none)'} | "
                        f"{decision.selected_model_identifier or '(model unknown)'} | "
                        f"{decision.fallback_reason or 'no fallback'}"
                    ),
                    "warning" if decision.used_fallback else "info",
                )
                for decision in registry.last_route_decisions[:6]
            )
            control_details = [
                "\n".join(
                    (
                        f"Installed registrations: {len(registry.registrations)}",
                        f"Active heavy roles: {', '.join(registry.active_heavy_roles) if registry.active_heavy_roles else '(none)'}",
                        f"Governor active: {self._bool_word(registry.governor_active)}",
                        f"Governor summary: {registry.governor_summary or '(inactive)'}",
                    )
                )
            ]
            control_details.extend(
                "\n".join(
                    (
                        f"Requested role: {decision.requested_role.value}",
                        f"Selected registration: {decision.selected_registration_id or '(none)'}",
                        f"Selected model: {decision.selected_model_identifier or '(unknown)'}",
                        f"Fallback reason: {decision.fallback_reason or '(none)'}",
                        f"Used fallback: {self._bool_word(decision.used_fallback)}",
                    )
                )
                for decision in registry.last_route_decisions[:6]
            )
            self._populate_list(self._control_plane_list, control_entries, details=control_details)
            self._control_plane_panel.set_summary(
                f"{len(registry.registrations)} registrations | {len(registry.last_route_decisions[:6])} recent route decisions"
            )
            self._control_plane_panel.set_empty_detail("Select a control-plane row to inspect routing and governor decisions.")

            runtime_entries = [
                ("Generation backend", health.generation_backend or "(unknown)", "info"),
                ("Embedding backend", health.embedding_backend or "(unknown)", "info"),
                ("Heavy slots", f"{len(health.active_heavy_roles)}/{health.heavy_slot_limit}", "accent"),
                ("Governor active", self._bool_word(health.governor_active), "warning" if health.governor_active else "muted"),
                ("Governor summary", health.governor_summary or "(none)", "warning" if health.governor_summary else "muted"),
                ("Fallback active", self._bool_word(health.fallback_active), "warning" if health.fallback_active else "muted"),
                ("Fallback reason", health.fallback_reason or "(none)", "warning" if health.fallback_reason else "muted"),
                ("Last error", health.last_error or "(none)", "danger" if health.last_error else "muted"),
            ]
            if shell_state.workspace_mode == "coding_workspace":
                runtime_entries.append(
                    (
                        "Coding status",
                        f"{coding_output.status or coding_practice.status or 'idle'} | {shell_state.quality_gate_state}",
                        "accent",
                    )
                )
            runtime_details = [
                "\n".join(
                    (
                        f"Generation backend: {health.generation_backend or '(unknown)'}",
                        f"Embedding backend: {health.embedding_backend or '(unknown)'}",
                        f"Heavy slots: {len(health.active_heavy_roles)}/{health.heavy_slot_limit}",
                        f"Governor active: {self._bool_word(health.governor_active)}",
                        f"Governor summary: {health.governor_summary or '(none)'}",
                        f"Fallback active: {self._bool_word(health.fallback_active)}",
                        f"Fallback reason: {health.fallback_reason or '(none)'}",
                        f"Last error: {health.last_error or '(none)'}",
                    )
                )
                for _headline, _detail, _tone in runtime_entries
            ]
            self._populate_list(self._runtime_list, runtime_entries, details=runtime_details)
            self._runtime_panel.set_summary(
                f"Heavy roles {len(health.active_heavy_roles)}/{health.heavy_slot_limit} | governor {'active' if health.governor_active else 'idle'}"
            )
            self._runtime_panel.set_empty_detail("Select a runtime row to inspect backend and pressure details.")

            practice_entries = [
                (
                    session_item.prompt or session_item.session_id or "Coding Dojo",
                    (
                        f"{session_item.status} | score {session_item.quality_score:.2f} | "
                        f"validated {len(session_item.validated_patterns)} | rejected {len(session_item.rejected_patterns)}"
                    ),
                    "accent" if session_item.status not in {"", "idle"} else "muted",
                )
                for session_item in recent_practice_sessions
            ] or [("Practice", "No coding practice log loaded.", "muted")]
            practice_details = [
                "\n".join(
                    (
                        f"Practice session: {session_item.session_id or '(none)'}",
                        f"Status: {session_item.status}",
                        f"Prompt: {session_item.prompt or '(none)'}",
                        f"Quality score: {session_item.quality_score:.2f}",
                        f"Validated patterns: {', '.join(session_item.validated_patterns) if session_item.validated_patterns else '(none)'}",
                        f"Rejected patterns: {', '.join(session_item.rejected_patterns) if session_item.rejected_patterns else '(none)'}",
                        f"Warnings: {', '.join(session_item.warnings) if session_item.warnings else '(none)'}",
                        f"Started: {_format_timestamp(session_item.started_at)}",
                        f"Completed: {_format_timestamp(session_item.completed_at)}",
                    )
                )
                for session_item in recent_practice_sessions
            ] or ["No coding practice log loaded."]
            self._populate_list(
                self._practice_log_list,
                practice_entries,
                details=practice_details,
            )
            recent_practice_scores = [float(session_item.quality_score) for session_item in recent_practice_sessions]
            self._practice_log_panel.set_summary(
                f"{len(recent_practice_sessions)} sessions | trend {self._score_trend_text(recent_practice_scores)}"
                if recent_practice_sessions
                else "No recent Coding Dojo sessions loaded."
            )
            self._practice_log_panel.set_empty_detail("Select a practice row to inspect the current coding-dojo session.")

            tier_buckets: dict[str, list[Any]] = {"verified": [], "candidate": [], "rejected": []}
            for pattern in app_state.coding_patterns:
                tier_buckets.setdefault(str(getattr(pattern.tier, "value", pattern.tier)), []).append(pattern)
            pattern_entries = [
                (
                    pattern.title or pattern.pattern_id,
                    (
                        f"{getattr(pattern.tier, 'value', pattern.tier)} | "
                        f"{pattern.language or 'n/a'} | score {pattern.quality_score:.2f} | reuse {pattern.reuse_count}"
                    ),
                    "success"
                    if str(getattr(pattern.tier, "value", pattern.tier)) == "verified"
                    else "warning"
                    if str(getattr(pattern.tier, "value", pattern.tier)) == "rejected"
                    else "info",
                )
                for pattern in app_state.coding_patterns[:24]
            ] or [("Patterns", "No indexed coding patterns loaded.", "muted")]
            pattern_details = [
                "\n".join(
                    (
                        f"Pattern: {pattern.title or pattern.pattern_id}",
                        f"Tier: {getattr(pattern.tier, 'value', pattern.tier)}",
                        f"Language: {pattern.language or '(n/a)'}",
                        f"Framework: {pattern.framework or '(n/a)'}",
                        f"Task type: {getattr(pattern.task_type, 'value', pattern.task_type)}",
                        f"Source: {pattern.source or '(none)'}",
                        f"Quality score: {pattern.quality_score:.2f}",
                        f"Reuse count: {pattern.reuse_count}",
                        f"Last used: {_format_timestamp(getattr(pattern, 'last_used_at', None))}",
                        f"Validation history: {len(pattern.validation_history)} entries",
                        "",
                        pattern.summary or pattern.code_snippet or "(no pattern summary stored)",
                    )
                )
                for pattern in app_state.coding_patterns[:24]
            ] or ["No indexed coding patterns loaded."]
            self._populate_list(self._coding_patterns_list, pattern_entries, details=pattern_details)
            self._coding_patterns_panel.set_summary(
                f"Indexed patterns {len(app_state.coding_patterns)} | verified {len(tier_buckets.get('verified', []))} | candidate {len(tier_buckets.get('candidate', []))} | rejected {len(tier_buckets.get('rejected', []))}"
            )
            self._coding_patterns_panel.set_empty_detail("Select a pattern to inspect its quality, tier, and validation history.")

            memory_entries: list[tuple[str, str, str]] = []
            memory_details: list[str] = []
            for tier_name, tone in (("verified", "success"), ("candidate", "info"), ("rejected", "warning")):
                bucket = tier_buckets.get(tier_name, [])
                top_pattern = max(bucket, key=lambda item: (item.reuse_count, item.quality_score), default=None)
                languages = sorted({pattern.language for pattern in bucket if pattern.language})
                frameworks = sorted({pattern.framework for pattern in bucket if pattern.framework})
                task_types = sorted(
                    {
                        str(getattr(getattr(pattern, "task_type", ""), "value", getattr(pattern, "task_type", "")))
                        for pattern in bucket
                        if getattr(pattern, "task_type", "")
                    }
                )
                latest_use = max((getattr(pattern, "last_used_at", None) for pattern in bucket), default=None)
                avg_quality = _mean([float(pattern.quality_score) for pattern in bucket]) if bucket else 0.0
                reuse_total = sum(int(pattern.reuse_count) for pattern in bucket)
                validation_total = sum(len(pattern.validation_history) for pattern in bucket)
                memory_entries.append(
                    (
                        tier_name.title(),
                        f"{len(bucket)} patterns | avg {avg_quality:.2f} | top {top_pattern.title if top_pattern else '(none)'}",
                        tone,
                    )
                )
                memory_details.append(
                    "\n".join(
                        (
                            f"Tier: {tier_name}",
                            f"Count: {len(bucket)}",
                            f"Languages: {', '.join(languages) if languages else '(none)'}",
                            f"Frameworks: {', '.join(frameworks) if frameworks else '(none)'}",
                            f"Task types: {', '.join(task_types) if task_types else '(none)'}",
                            f"Average quality: {avg_quality:.2f}",
                            f"Reuse total: {reuse_total}",
                            f"Validation history entries: {validation_total}",
                            f"Last used: {_format_timestamp(latest_use)}",
                            "Entries: "
                            + (
                                " | ".join(
                                    f"{pattern.title or pattern.pattern_id} "
                                    f"(score {pattern.quality_score:.2f}, reuse {pattern.reuse_count}, last {_format_timestamp(getattr(pattern, 'last_used_at', None))})"
                                    for pattern in bucket[:6]
                                )
                                if bucket
                                else "(none)"
                            ),
                        )
                    )
                )
            routed_memory = [
                f"verified {', '.join(coding_output.verified_patterns) or '(none)'}",
                f"candidate {', '.join(coding_output.candidate_patterns) or '(none)'}",
                f"rejected {', '.join(coding_output.rejected_patterns) or '(none)'}",
            ]
            memory_entries.append(("Current task routing memory", " | ".join(routed_memory), "accent"))
            memory_details.append(
                "\n".join(
                    (
                        f"Current request: {coding_output.request_id or '(none)'}",
                        f"Verified patterns: {', '.join(coding_output.verified_patterns) or '(none)'}",
                        f"Candidate patterns: {', '.join(coding_output.candidate_patterns) or '(none)'}",
                        f"Rejected patterns: {', '.join(coding_output.rejected_patterns) or '(none)'}",
                        f"Recent coding runs: {len(recent_coding_results)}",
                        f"Recent practice sessions: {len(recent_practice_sessions)}",
                    )
                )
            )
            self._populate_list(self._coding_memory_list, memory_entries, details=memory_details)
            self._coding_memory_panel.set_summary(
                f"Memory tiers | verified {len(tier_buckets.get('verified', []))} | candidate {len(tier_buckets.get('candidate', []))} | rejected {len(tier_buckets.get('rejected', []))} | runs {len(recent_coding_results)}"
            )
            self._coding_memory_panel.set_empty_detail("Select a memory tier to inspect coding-pattern promotion state.")

            validation_entries = [
                (
                    "Active report",
                    (
                        f"quality {coding_output.quality_report.quality_score:.2f} | "
                        f"tests={self._bool_word(coding_output.quality_report.tests_passed)} | "
                        f"lint={self._bool_word(coding_output.quality_report.lint_passed)} | "
                        f"security={self._bool_word(coding_output.quality_report.security_passed)} | "
                        f"regression={self._bool_word(coding_output.quality_report.regression_passed)}"
                    ),
                    "success" if coding_output.quality_report.overall_passed else "warning",
                )
            ]
            limited_validation_warnings = sorted(
                {
                    warning
                    for result in recent_coding_results
                    for warning in tuple(result.warnings)
                    if any(token in warning for token in ("skipped", "unavailable", "limited", "blocked"))
                }
            )
            for pattern in app_state.coding_patterns[:8]:
                if not pattern.validation_history:
                    continue
                latest = pattern.validation_history[-1]
                validation_entries.append(
                    (
                        pattern.title or pattern.pattern_id,
                        (
                            f"pass {', '.join(latest.checks_passed) or '(none)'} | "
                            f"fail {', '.join(latest.checks_failed) or '(none)'}"
                        ),
                        "warning" if latest.checks_failed else "success",
                    )
                )
            if limited_validation_warnings:
                validation_entries.append(
                    (
                        "History limits",
                        " | ".join(limited_validation_warnings[:3]),
                        "warning",
                    )
                )
            if coding_output.warnings:
                validation_entries.extend(
                    ("Coding warning", warning, "warning") for warning in coding_output.warnings[:4]
                )
            validation_details = [
                "\n".join(
                    (
                        f"Quality score: {coding_output.quality_report.quality_score:.2f}",
                        f"Overall passed: {self._bool_word(coding_output.quality_report.overall_passed)}",
                        f"Tests: {self._bool_word(coding_output.quality_report.tests_passed)}",
                        f"Lint: {self._bool_word(coding_output.quality_report.lint_passed)}",
                        f"Security: {self._bool_word(coding_output.quality_report.security_passed)}",
                        f"Regression: {self._bool_word(coding_output.quality_report.regression_passed)}",
                    )
                )
            ]
            for pattern in app_state.coding_patterns[:8]:
                if not pattern.validation_history:
                    continue
                latest = pattern.validation_history[-1]
                validation_details.append(
                    "\n".join(
                        (
                            f"Pattern: {pattern.title or pattern.pattern_id}",
                            f"Validation id: {latest.validation_id or '(none)'}",
                            f"Checks passed: {', '.join(latest.checks_passed) if latest.checks_passed else '(none)'}",
                            f"Checks failed: {', '.join(latest.checks_failed) if latest.checks_failed else '(none)'}",
                            f"Reviewer summary: {latest.reviewer_summary or '(none)'}",
                        )
                    )
                )
            if limited_validation_warnings:
                validation_details.append(
                    "\n".join(
                        (
                            "Recent validation limits:",
                            *[f"- {warning}" for warning in limited_validation_warnings[:6]],
                        )
                    )
                )
            validation_details.extend(f"Warning: {warning}" for warning in coding_output.warnings[:4])
            self._populate_list(self._coding_validation_list, validation_entries, details=validation_details)
            failed_gate_count = sum(
                1
                for passed in (
                    coding_output.quality_report.tests_passed,
                    coding_output.quality_report.lint_passed,
                    coding_output.quality_report.complexity_passed,
                    coding_output.quality_report.security_passed,
                    coding_output.quality_report.maintainability_passed,
                    coding_output.quality_report.critique_passed,
                    coding_output.quality_report.regression_passed,
                )
                if not passed
            )
            self._coding_validation_panel.set_summary(
                f"Active quality {coding_output.quality_report.quality_score:.2f} | failed gates {failed_gate_count} | limits {len(limited_validation_warnings)}"
            )
            self._coding_validation_panel.set_empty_detail("Select a validation row to inspect gate and reviewer details.")

            passed_gates = sum(
                1
                for passed in (
                    coding_output.quality_report.tests_passed,
                    coding_output.quality_report.lint_passed,
                    coding_output.quality_report.complexity_passed,
                    coding_output.quality_report.security_passed,
                    coding_output.quality_report.maintainability_passed,
                    coding_output.quality_report.critique_passed,
                    coding_output.quality_report.regression_passed,
                )
                if passed
            )
            pattern_reuse_total = sum(pattern.reuse_count for pattern in app_state.coding_patterns)
            top_reuse_pattern = max(app_state.coding_patterns, key=lambda item: item.reuse_count, default=None)
            route_assignments = coding_output.role_assignments or {}
            route_summary_text = " | ".join(coding_output.route_summary or shell_state.active_route_summary) or "(none)"
            overall_pass_count = sum(1 for result in recent_coding_results if result.quality_report.overall_passed)
            lint_fail_count = sum(1 for result in recent_coding_results if not result.quality_report.lint_passed)
            security_fail_count = sum(1 for result in recent_coding_results if not result.quality_report.security_passed)
            regression_fail_count = sum(1 for result in recent_coding_results if not result.quality_report.regression_passed)
            bug_fix_results = [
                result
                for result in recent_coding_results
                if str(getattr(getattr(result, "task_type", ""), "value", getattr(result, "task_type", ""))) == "bug_fixing"
            ]
            bug_fix_success_count = sum(1 for result in bug_fix_results if result.quality_report.overall_passed)
            practice_scores = [float(session_item.quality_score) for session_item in recent_practice_sessions]
            memory_growth_text = (
                f"{len(app_state.coding_patterns)} indexed | verified {len(tier_buckets.get('verified', []))} | candidate {len(tier_buckets.get('candidate', []))}"
            )
            model_performance_lines = self._model_performance_lines(recent_coding_results)
            average_recent_quality = _mean(
                [float(result.quality_report.quality_score) for result in recent_coding_results]
            ) if recent_coding_results else 0.0
            metrics_entries = [
                (
                    "Pass / fail rate",
                    f"{overall_pass_count}/{len(recent_coding_results)} passed | avg quality {average_recent_quality:.2f}",
                    "success" if recent_coding_results and overall_pass_count == len(recent_coding_results) else "warning",
                ),
                (
                    "Lint / static checks",
                    f"lint failed {lint_fail_count} | security failed {security_fail_count}",
                    "warning" if lint_fail_count or security_fail_count else "success",
                ),
                (
                    "Bug-fix success",
                    f"{bug_fix_success_count}/{len(bug_fix_results)} bug-fix runs passed",
                    "success" if bug_fix_results and bug_fix_success_count == len(bug_fix_results) else "muted",
                ),
                (
                    "Regression rate",
                    f"{regression_fail_count}/{len(recent_coding_results)} runs failed regression",
                    "warning" if regression_fail_count else "success",
                ),
                (
                    "Practice score trend",
                    self._score_trend_text(practice_scores),
                    "accent" if practice_scores else "muted",
                ),
                (
                    "Pattern memory growth",
                    memory_growth_text,
                    "info",
                ),
                (
                    "Pattern reuse score",
                    f"total {pattern_reuse_total} | top {top_reuse_pattern.title if top_reuse_pattern else '(none)'}",
                    "accent" if pattern_reuse_total else "muted",
                ),
                (
                    "Per-model quality",
                    model_performance_lines[0],
                    "accent" if route_assignments or route_summary_text != "(none)" else "muted",
                ),
                (
                    "Validation caveats",
                    " | ".join((*coding_output.warnings[:2], *limited_validation_warnings[:1])) or (shell_state.degraded_reason or "(none)"),
                    "warning" if coding_output.warnings or shell_state.degraded_reason or limited_validation_warnings else "muted",
                ),
            ]
            metrics_details = [
                "\n".join(
                    (
                        f"Recent runs: {len(recent_coding_results)}",
                        f"Passed: {overall_pass_count}",
                        f"Failed: {len(recent_coding_results) - overall_pass_count}",
                        f"Average recent quality: {average_recent_quality:.2f}",
                        "Recent scores: "
                        + (
                            ", ".join(f"{result.quality_report.quality_score:.2f}" for result in recent_coding_results[:6])
                            if recent_coding_results
                            else "(none)"
                        ),
                    )
                ),
                "\n".join(
                    (
                        f"Lint failed runs: {lint_fail_count}",
                        f"Security failed runs: {security_fail_count}",
                        "Static check history: "
                        + (
                            " | ".join(
                                f"{result.request_id or result.task_type}:{'lint' if not result.quality_report.lint_passed else 'ok'}/"
                                f"{'security' if not result.quality_report.security_passed else 'ok'}"
                                for result in recent_coding_results[:6]
                            )
                            if recent_coding_results
                            else "(none)"
                        ),
                    )
                ),
                "\n".join(
                    (
                        f"Bug-fix runs: {len(bug_fix_results)}",
                        f"Bug-fix passed: {bug_fix_success_count}",
                        "Bug-fix requests: "
                        + (
                            ", ".join(result.request_id or "(none)" for result in bug_fix_results[:6])
                            if bug_fix_results
                            else "(none)"
                        ),
                    )
                ),
                "\n".join(
                    (
                        f"Regression failed runs: {regression_fail_count}",
                        "Regression history: "
                        + (
                            " | ".join(
                                f"{result.request_id or result.task_type}:{self._bool_word(result.quality_report.regression_passed)}"
                                for result in recent_coding_results[:6]
                            )
                            if recent_coding_results
                            else "(none)"
                        ),
                    )
                ),
                "\n".join(
                    (
                        f"Recent practice sessions: {len(recent_practice_sessions)}",
                        f"Practice trend: {self._score_trend_text(practice_scores)}",
                        "Practice scores: "
                        + (
                            ", ".join(f"{score:.2f}" for score in practice_scores[:6])
                            if practice_scores
                            else "(none)"
                        ),
                    )
                ),
                "\n".join(
                    (
                        f"Pattern reuse total: {pattern_reuse_total}",
                        f"Average reuse: {(pattern_reuse_total / max(1, len(app_state.coding_patterns))):.2f}",
                        f"Top reused pattern: {top_reuse_pattern.title if top_reuse_pattern else '(none)'}",
                        f"Current shell counts: {shell_state.pattern_tier_counts or {}}",
                    )
                ),
                "\n".join(
                    (
                        f"Memory growth: {memory_growth_text}",
                        "Role assignments: "
                        + (
                            " | ".join(f"{role}={model}" for role, model in route_assignments.items())
                            if route_assignments
                            else "(none)"
                        ),
                        f"Route summary: {route_summary_text}",
                        f"Current coding state: {shell_state.coding_state or '(idle)'}",
                    )
                ),
                "\n".join(model_performance_lines),
                "\n".join(
                    (
                        f"Warnings: {', '.join(coding_output.warnings) if coding_output.warnings else '(none)'}",
                        f"Historical limits: {', '.join(limited_validation_warnings) if limited_validation_warnings else '(none)'}",
                        f"Degraded reason: {shell_state.degraded_reason or '(none)'}",
                        f"Sandbox: {shell_state.sandbox_state}",
                        f"Quality gates: {shell_state.quality_gate_state}",
                    )
                ),
            ]
            self._populate_list(self._coding_metrics_list, metrics_entries, details=metrics_details)
            self._coding_metrics_panel.set_summary(
                f"Quality {coding_output.quality_report.quality_score:.2f} | pass {overall_pass_count}/{len(recent_coding_results)} | practice {self._score_trend_text(practice_scores)}"
                if recent_coding_results
                else f"Quality {coding_output.quality_report.quality_score:.2f} | reuse total {pattern_reuse_total} | route roles {len(route_assignments)}"
            )
            self._coding_metrics_panel.set_empty_detail("Select a metrics row to inspect coding quality and memory summaries.")
            self._refresh_coding_route_controls()

        def _refresh_history_details(self, *_args: Any) -> None:
            selected_id = ""
            current_item = self._history_list.currentItem()
            if current_item is not None:
                selected_id = str(current_item.data(QtCore.Qt.ItemDataRole.UserRole) or "")
            selected_task = self._app_state.selected_task
            if selected_task.task_id and (not selected_id or selected_task.task_id == selected_id):
                lines = [
                    f"Task: {selected_task.task_id}",
                    f"Question: {selected_task.question or '(none)'}",
                    f"Verifier: {selected_task.selected_verifier or '(none)'}",
                    f"Critique: {selected_task.critique_result or '(none)'}",
                    f"Candidates: {selected_task.candidate_trace_count}",
                    f"Warnings: {', '.join(selected_task.warnings) if selected_task.warnings else '(none)'}",
                    f"Citations: {', '.join(selected_task.citation_refs) if selected_task.citation_refs else '(none)'}",
                    f"Repairs: {', '.join(selected_task.repair_actions) if selected_task.repair_actions else '(none)'}",
                    f"Failures: {', '.join(selected_task.failure_categories) if selected_task.failure_categories else '(none)'}",
                    f"Optimizer lifecycle: {', '.join(selected_task.optimizer_lifecycle) if selected_task.optimizer_lifecycle else '(none)'}",
                    f"Advisor summaries: {', '.join(selected_task.advisor_summaries) if selected_task.advisor_summaries else '(none)'}",
                    f"Trace/timeline export: {selected_task.trace_debug_export_path or '(none)'}",
                    "",
                    selected_task.answer_text or "(no answer text loaded)",
                ]
                self._history_detail.setPlainText("\n".join(lines))
                return
            entry = next((item for item in self._app_state.task_history if item.task_id == selected_id), None)
            if entry is None and self._app_state.task_history:
                entry = self._app_state.task_history[0]
            if entry is None:
                self._history_detail.setPlainText("No task history loaded.")
                return
            self._history_detail.setPlainText(
                "\n".join(
                    (
                        f"Task: {entry.task_id}",
                        f"Question: {entry.question}",
                        f"Critique: {entry.critique_result or '(none)'}",
                        f"Strategy: {entry.selected_strategy or '(none)'}",
                        f"Verifier: {entry.selected_verifier or '(none)'}",
                        f"Candidates: {entry.candidate_trace_count}",
                        f"Citations: {entry.citation_count}",
                        f"Warnings: {entry.warning_count}",
                        f"Degraded: {entry.degraded_reason or '(none)'}",
                        f"Web fallback: {self._bool_word(entry.used_web_fallback)}",
                        "",
                        entry.answer_preview or "(no answer preview)",
                    )
                )
            )

        def _refresh_bottom_surfaces(self) -> None:
            history_entries = [
                (
                    entry.question,
                    (
                        f"{entry.critique_result or 'pending'} | candidates {entry.candidate_trace_count} | "
                        f"citations {entry.citation_count} | warnings {entry.warning_count}"
                    ),
                    "warning" if entry.degraded_reason else "info",
                )
                for entry in self._app_state.task_history[:20]
            ] or [("History", "No task history loaded.", "muted")]
            history_ids = [entry.task_id for entry in self._app_state.task_history[:20]]
            self._populate_list(self._history_list, history_entries, user_data=history_ids)
            self._refresh_history_details()

            knowledge_entries = [
                (
                    source.title or source.source_ref,
                    f"{source.corpus_origin or '(local)'} | chunks {source.chunk_count} | archived={source.archived}",
                    "warning" if source.archived else "info",
                )
                for source in self._app_state.knowledge_sources[:40]
            ] or [("Knowledge", "No knowledge sources loaded.", "muted")]
            self._populate_list(self._knowledge_list, knowledge_entries)

        def _coding_debug_lines(self) -> list[str]:
            app_state = self._app_state
            coding_result = app_state.coding_output
            practice_result = app_state.coding_practice.task_result
            artifact_pool: list[Any] = []
            artifact_pool.extend(self._coding_debug_artifacts(coding_result.artifacts))
            if practice_result.status not in {"", "idle"}:
                artifact_pool.extend(self._coding_debug_artifacts(practice_result.artifacts))
            debug_lines = [
                "",
                "coding_debug:",
                "Primary coding cards intentionally exclude raw sandbox traces.",
                f"current_request: {coding_result.request_id or practice_result.request_id or '(none)'}",
                f"quality_score: {coding_result.quality_report.quality_score:.2f}",
                f"sandbox_state: {self._shell_state.sandbox_state}",
            ]
            if not artifact_pool:
                debug_lines.append("No secondary sandbox trace/stdout/stderr artifacts are currently stored.")
                return debug_lines
            for artifact in artifact_pool[:4]:
                debug_lines.extend(
                    (
                        "",
                        f"artifact: {artifact.title or artifact.artifact_type}",
                        f"type: {artifact.artifact_type}",
                        f"path: {artifact.path or '(none)'}",
                        "preview:",
                        artifact.content_preview or "(empty preview)",
                    )
                )
                metadata = dict(getattr(artifact, "metadata", {}) or {})
                stdout_preview = str(metadata.get("stdout_preview", "")).strip()
                stderr_preview = str(metadata.get("stderr_preview", "")).strip()
                trace_preview = str(metadata.get("trace_preview", "")).strip()
                if stdout_preview:
                    debug_lines.extend(("stdout_preview:", stdout_preview))
                if stderr_preview:
                    debug_lines.extend(("stderr_preview:", stderr_preview))
                if trace_preview:
                    debug_lines.extend(("trace_preview:", trace_preview))
            return debug_lines

        def _refresh_center_cards(self) -> None:
            shell_state = self._shell_state
            app_state = self._app_state
            task = app_state.active_task
            coding_output = app_state.coding_output
            coding_practice = app_state.coding_practice
            is_coding = shell_state.workspace_mode == "coding_workspace"
            coding_result = coding_output
            if (
                coding_result.status == "idle"
                and not coding_result.summary
                and not coding_result.artifacts
                and coding_practice.task_result.status != "idle"
            ):
                coding_result = coding_practice.task_result

            stage_text = (
                coding_result.active_phase
                if is_coding and coding_result.active_phase not in {"", "idle"}
                else task.running_stage or app_state.last_stage or shell_state.coding_state
            )
            active_summary = (
                coding_result.prompt
                or coding_practice.prompt
                or task.question
                or shell_state.current_task_summary
                or shell_state.sub_status_text
            )
            active_metrics: list[str] = []
            if stage_text:
                active_metrics.append(f"Phase {self._format_stage_label(stage_text)}")
            if shell_state.elapsed_seconds > 0:
                active_metrics.append(f"Elapsed {self._format_elapsed(shell_state.elapsed_seconds)}")
            active_metrics.append(f"Candidates {shell_state.candidate_count}")
            active_metrics.append(f"Evidence {shell_state.evidence_count}")
            active_metrics.append(f"Verifier {shell_state.verifier_state or 'idle'}")
            active_metrics.append(f"Confidence {shell_state.confidence_band or 'low'}")
            if is_coding:
                active_metrics.append(f"Sandbox {shell_state.sandbox_state}")
                active_metrics.append(f"Gates {shell_state.quality_gate_state}")
                if shell_state.current_file:
                    active_metrics.append(f"File {shell_state.current_file}")
                if shell_state.current_project:
                    active_metrics.append(f"Project {shell_state.current_project}")
            routes_text = (
                "Routes: " + " | ".join(shell_state.active_route_summary[:4])
                if shell_state.active_route_summary
                else (
                    "Active roles: " + ", ".join(shell_state.active_model_roles[:4])
                    if shell_state.active_model_roles
                    else ""
                )
            )
            role_lines: list[str] = []
            if shell_state.active_model_roles:
                role_lines.append("Active roles: " + ", ".join(shell_state.active_model_roles[:6]))
            if shell_state.active_agent:
                role_lines.append(
                    "Lead: " + str(shell_state.active_agent).replace("_", " ").replace(".", " ").title()
                )
            if shell_state.secondary_agents:
                role_lines.append(
                    "Support: "
                    + ", ".join(
                        str(agent).replace("_", " ").replace(".", " ").title()
                        for agent in shell_state.secondary_agents[:5]
                    )
                )
            if shell_state.active_tools:
                role_lines.append("Tools: " + ", ".join(str(tool).replace("_", " ") for tool in shell_state.active_tools[:6]))
            warning_parts = []
            if shell_state.fallback_reason:
                warning_parts.append(f"Fallback: {shell_state.fallback_reason}")
            if shell_state.degraded_reason:
                warning_parts.append(f"Degraded: {shell_state.degraded_reason}")
            if shell_state.approval_pending and shell_state.approval_prompt_summary:
                warning_parts.append(f"Approval: {shell_state.approval_prompt_summary}")
            if app_state.last_notice and app_state.last_notice_severity in {"warning", "error"}:
                warning_parts.append(f"{app_state.last_notice_severity.title()}: {app_state.last_notice}")
            if shell_state.confidence_band in {"low", "medium"} and bool(task.answer_text.strip()):
                warning_parts.append(f"Uncertainty: {shell_state.confidence_band} confidence")
            if shell_state.optimizer_state not in {"", "idle", "inactive"}:
                warning_parts.append(f"Optimizer: {shell_state.optimizer_state}")
            if task.advisor_summaries:
                warning_parts.append("Advice: " + " | ".join(task.advisor_summaries[:2]))
            show_active = bool(active_summary.strip()) and any(
                (
                    self._looks_active_state(stage_text),
                    self._looks_active_state(shell_state.active_agent),
                    self._looks_active_state(shell_state.coding_state),
                    self._looks_active_state(shell_state.long_horizon_state),
                    self._looks_active_state(shell_state.capability_session_state),
                    self._looks_active_state(app_state.local_task_session.status),
                    shell_state.approval_pending,
                )
            )
            self._active_task_card.setVisible(show_active)
            self._active_task_title.setText("Active Coding Task" if is_coding else "Active Task")
            self._active_task_phase.setText(self._format_stage_label(stage_text))
            self._active_task_summary.setText(active_summary)
            self._active_task_metrics.setText("   |   ".join(active_metrics))
            self._active_task_routes.setText(routes_text)
            self._active_route_block.setVisible(bool(routes_text))
            self._active_task_roles.setText("\n".join(role_lines[:3]))
            self._active_roles_block.setVisible(bool(role_lines))
            self._active_task_warnings.setText("\n".join(warning_parts[:4]))
            self._active_warning_block.setVisible(bool(warning_parts))
            self._active_task_detail_stack.setVisible(
                self._active_route_block.isVisible()
                or self._active_roles_block.isVisible()
                or self._active_warning_block.isVisible()
            )
            previous_operator_expanded = self._operator_expanded_mode
            self._operator_expanded_mode = show_active and bool(
                shell_state.fallback_reason
                or shell_state.degraded_reason
                or shell_state.approval_pending
                or app_state.last_notice_severity in {"warning", "error"}
                or shell_state.capability_session_state not in {"", "inactive"}
            )
            if self._operator_expanded_mode != previous_operator_expanded:
                self._center_splitter_user_override = False

            coding_card_title = "Coding Workspace"
            if shell_state.coding_state == "testing":
                coding_card_title = "Sandbox And Validation"
            elif shell_state.coding_state in {"reviewing", "refactoring"}:
                coding_card_title = "Review And Refactor"
            elif shell_state.coding_state == "generating":
                coding_card_title = "Patch Workspace"
            validator_label = (
                "quality_gates"
                if shell_state.quality_gate_state not in {"", "idle", "passed"}
                else task.selected_verifier
                or shell_state.verifier_state
                or "local_validator"
            )
            coding_role_summary = (
                ", ".join(
                    f"{str(role).replace('_', ' ')}={model}"
                    for role, model in coding_result.role_assignments.items()
                )
                if coding_result.role_assignments
                else "(no role assignments)"
            )
            context_lines = [
                f"Task: {getattr(coding_result.task_type, 'value', coding_result.task_type)}",
                f"Project: {shell_state.current_project or '(none)'}",
                f"File: {shell_state.current_file or '(none)'}",
                f"Validator: {validator_label}",
                f"Roles: {coding_role_summary}",
                f"Routes: {' | '.join(shell_state.active_route_summary) or '(none)'}",
            ]
            validation_lines = [
                f"Sandbox {shell_state.sandbox_state}",
                f"Gates {shell_state.quality_gate_state}",
                (
                    "Checks "
                    f"tests={self._bool_word(coding_result.quality_report.tests_passed)} "
                    f"lint={self._bool_word(coding_result.quality_report.lint_passed)} "
                    f"security={self._bool_word(coding_result.quality_report.security_passed)} "
                    f"regression={self._bool_word(coding_result.quality_report.regression_passed)}"
                ),
            ]
            primary_artifacts = self._primary_coding_artifacts(coding_result.artifacts)
            artifact_lines = [
                f"{artifact.title or artifact.artifact_type}: {artifact.path or artifact.content_preview[:60]}"
                for artifact in primary_artifacts[:4]
            ]
            pattern_hint = (
                "Pattern reuse hint: "
                f"V{shell_state.pattern_tier_counts.get('verified', 0)} "
                f"C{shell_state.pattern_tier_counts.get('candidate', 0)} "
                f"R{shell_state.pattern_tier_counts.get('rejected', 0)}"
            )
            blocked_gates: list[str] = []
            report = coding_result.quality_report
            if coding_result.status not in {"", "idle"}:
                if not report.tests_passed:
                    blocked_gates.append("tests")
                if not report.lint_passed:
                    blocked_gates.append("lint")
                if not report.security_passed:
                    blocked_gates.append("security")
                if not report.regression_passed:
                    blocked_gates.append("regression")
                if not report.maintainability_passed:
                    blocked_gates.append("maintainability")
                if not report.complexity_passed:
                    blocked_gates.append("complexity")
            if shell_state.quality_gate_state not in {"", "idle", "passed"}:
                blocked_gates.append(shell_state.quality_gate_state)
            if shell_state.degraded_reason:
                blocked_gates.append(shell_state.degraded_reason)
            if coding_result.warnings:
                blocked_gates.extend(coding_result.warnings[:2])
            if shell_state.sandbox_state == "disabled":
                blocked_gates.append("sandbox disabled by profile")
            if shell_state.sandbox_state == "skipped":
                blocked_gates.append("sandbox execution limited in this environment")
            for route_entry in shell_state.active_route_summary:
                if "fallback" in route_entry.lower():
                    blocked_gates.append(route_entry)
            show_coding_card = is_coding and any(
                (
                    active_summary.strip(),
                    artifact_lines,
                    self._looks_active_state(coding_result.status),
                    self._looks_active_state(coding_practice.status),
                    shell_state.current_file,
                )
            )
            self._coding_workspace_card.setVisible(show_coding_card)
            self._coding_workspace_title.setText(coding_card_title)
            self._coding_workspace_status.setText((shell_state.coding_state or coding_result.status or "idle").replace("_", " ").title())
            self._coding_workspace_summary.setText(coding_result.prompt or coding_practice.prompt or active_summary)
            self._coding_workspace_context.setText("   |   ".join(context_lines))
            self._coding_workspace_validation.setText("   |   ".join(validation_lines + [pattern_hint]))
            self._coding_workspace_artifacts.setText(
                "Artifacts: " + (" | ".join(artifact_lines) if artifact_lines else "(none yet)")
            )
            self._coding_workspace_blockers.setText(
                "Blocked gates: " + (" | ".join(dict.fromkeys(blocked_gates)) if blocked_gates else "(none)")
            )

            final_body = (
                coding_result.summary
                or coding_practice.summary
                if is_coding
                else task.answer_text
            ) or (task.answer_text if not is_coding else "")
            references_text = ""
            evidence_text = ""
            warning_text = ""
            why_text = ""
            how_text = ""
            deep_text = ""
            if is_coding:
                final_title = "Validated Practice Output" if coding_practice.status != "idle" and coding_output.status == "idle" else "Validated Coding Output"
                final_status = (shell_state.quality_gate_state or coding_result.status or "idle").replace("_", " ").title()
                evidence_text = (
                    f"Validation Summary: quality {coding_result.quality_report.quality_score:.2f} | "
                    f"sandbox {shell_state.sandbox_state} | gates {shell_state.quality_gate_state} | "
                    f"patterns V{shell_state.pattern_tier_counts.get('verified', 0)} "
                    f"C{shell_state.pattern_tier_counts.get('candidate', 0)} "
                    f"R{shell_state.pattern_tier_counts.get('rejected', 0)}"
                )
                artifact_lines = [
                    f"{artifact.title or artifact.artifact_type}: {artifact.path or artifact.content_preview[:48]}"
                    for artifact in primary_artifacts[:4]
                ]
                references_text = "Artifacts: " + (" | ".join(artifact_lines) if artifact_lines else "(none)")
                warning_values = list(coding_result.warnings) + list(coding_practice.warnings)
                if shell_state.fallback_reason:
                    warning_values.append(shell_state.fallback_reason)
                if shell_state.degraded_reason:
                    warning_values.append(shell_state.degraded_reason)
                if shell_state.quality_gate_state not in {"idle", "passed"}:
                    warning_values.append(f"quality_gates_{shell_state.quality_gate_state}")
                if shell_state.sandbox_state in {"disabled", "skipped", "failed"}:
                    warning_values.append(f"sandbox_{shell_state.sandbox_state}")
                warning_text = "Warnings: " + (" | ".join(warning_values[:4]) if warning_values else "(none)")
                why_text = "\n".join(
                    (
                        f"Task type: {getattr(coding_result.task_type, 'value', coding_result.task_type)}",
                        f"Route summary: {' | '.join(shell_state.active_route_summary) or '(none)'}",
                        f"Current project: {shell_state.current_project or '(none)'}",
                    )
                )
                how_text = "\n".join(
                    (
                        f"Sandbox: {shell_state.sandbox_state}",
                        f"Quality gates: {shell_state.quality_gate_state}",
                        (
                            "Checks: "
                            f"tests={'yes' if coding_result.quality_report.tests_passed else 'no'}, "
                            f"lint={'yes' if coding_result.quality_report.lint_passed else 'no'}, "
                            f"complexity={'yes' if coding_result.quality_report.complexity_passed else 'no'}, "
                            f"security={'yes' if coding_result.quality_report.security_passed else 'no'}, "
                            f"maintainability={'yes' if coding_result.quality_report.maintainability_passed else 'no'}, "
                            f"regression={'yes' if coding_result.quality_report.regression_passed else 'no'}"
                        ),
                    )
                )
                deep_bits = []
                if shell_state.practice_session_state != "idle":
                    deep_bits.append(f"Practice session: {shell_state.practice_session_state}")
                if shell_state.active_route_summary:
                    deep_bits.append(f"Routed roles: {' | '.join(shell_state.active_route_summary)}")
                if shell_state.pattern_tier_counts:
                    deep_bits.append(
                        "Pattern memory: "
                        f"V{shell_state.pattern_tier_counts.get('verified', 0)} "
                        f"C{shell_state.pattern_tier_counts.get('candidate', 0)} "
                        f"R{shell_state.pattern_tier_counts.get('rejected', 0)}"
                    )
                deep_text = "\n".join(deep_bits) or "No coding-depth deltas were recorded for this run."
                self._copy_answer_button.setText("Copy Summary")
                self._copy_citations_button.setText("Copy Artifacts")
            else:
                final_title = "Final Answer"
                final_status = (task.critique_result or shell_state.verifier_state or "pending").replace("_", " ").title()
                evidence_text = (
                    f"Evidence Summary: {shell_state.evidence_count} sources"
                    f" ({task.local_result_count} local / {task.web_result_count} web)"
                )
                references_text = "Citations: " + (" | ".join(task.citation_refs[:6]) if task.citation_refs else "(none)")
                warning_bits = []
                if shell_state.fallback_reason:
                    warning_bits.append(f"Fallback {shell_state.fallback_reason}")
                if shell_state.degraded_reason:
                    warning_bits.append(f"Degraded {shell_state.degraded_reason}")
                if shell_state.confidence_band in {"low", "medium"}:
                    warning_bits.append(f"Uncertainty {shell_state.confidence_band} confidence")
                if app_state.last_notice and app_state.last_notice_severity in {"warning", "error"}:
                    warning_bits.append(app_state.last_notice)
                warning_text = "Warnings: " + (" | ".join(warning_bits[:4]) if warning_bits else "(none)")
                why_text = "\n".join(
                    (
                        f"Selected verifier: {task.selected_verifier or shell_state.verifier_state or '(none)'}",
                        f"Evidence used: {shell_state.evidence_count}",
                        f"Route summary: {' | '.join(shell_state.active_route_summary) or '(none)'}",
                    )
                )
                how_text = "\n".join(
                    (
                        f"Verification result: {task.critique_result or shell_state.verifier_state or '(none)'}",
                        f"Repairs: {', '.join(task.repair_actions) if task.repair_actions else '(none)'}",
                        f"Failures avoided: {', '.join(task.failure_categories) if task.failure_categories else '(none)'}",
                    )
                )
                deep_bits = []
                if shell_state.candidate_count > 1:
                    deep_bits.append(f"Compared {shell_state.candidate_count} candidates.")
                if task.long_horizon_session_id:
                    deep_bits.append(
                        f"Long-horizon cycles: {task.long_horizon_completed_cycles}/{task.long_horizon_total_cycles or '?'}."
                    )
                if task.long_horizon_confidence_gain:
                    deep_bits.append(f"Confidence gain: {task.long_horizon_confidence_gain:.2f}.")
                if task.long_horizon_additional_supporting_evidence_count:
                    deep_bits.append(
                        f"Additional evidence: {task.long_horizon_additional_supporting_evidence_count}."
                    )
                deep_text = "\n".join(deep_bits) or "No deep-mode deltas were recorded for this run."
                self._copy_answer_button.setText("Copy Answer")
                self._copy_citations_button.setText("Copy Citations")

            show_final = bool(str(final_body).strip()) or (
                is_coding
                and (
                    bool(coding_result.artifacts)
                    or self._looks_active_state(coding_result.status)
                    or self._looks_active_state(coding_practice.status)
                )
            )
            self._final_answer_card.setVisible(show_final)
            self._final_answer_title.setText(final_title)
            self._final_answer_status.setText(final_status)
            self._final_answer_body.setText(final_body)
            self._final_answer_evidence.setText(evidence_text)
            self._final_answer_refs.setText(references_text)
            self._final_answer_warnings.setText(warning_text)
            self._why_answer_text.setPlainText(why_text)
            self._how_verified_text.setPlainText(how_text)
            self._deep_mode_text.setPlainText(deep_text)
            self._copy_answer_button.setEnabled(show_final)
            self._copy_citations_button.setEnabled(show_final)
            self._update_center_splitter_layout(force=self._operator_expanded_mode != previous_operator_expanded)

        def _message_card(self, role: str, title: str, body: str, chips: tuple[str, ...]) -> QtWidgets.QWidget:
            theme = self._current_theme()
            wrapper = QtWidgets.QWidget(self._conversation_container)
            row = QtWidgets.QHBoxLayout(wrapper)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(0)
            card = QtWidgets.QFrame(wrapper)
            card.setMaximumWidth(900)
            rail_color = (
                theme["accent"]
                if role == "user"
                else theme["success"]
                if role == "assistant"
                else theme["warning"]
            )
            base_color = (
                "rgba(18, 34, 58, 0.84)"
                if role == "user"
                else theme["card_final"]
                if role == "assistant"
                else theme["card_warning"]
            )
            card.setStyleSheet(
                "QFrame {"
                f"background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 {base_color}, stop:1 rgba(8, 16, 28, 0.90));"
                f"border: 1px solid {theme['edge']}; border-top: 2px solid {rail_color}; border-radius: 22px;"
                "}"
            )
            layout = QtWidgets.QVBoxLayout(card)
            layout.setContentsMargins(18, 16, 18, 16)
            layout.setSpacing(8)
            title_label = QtWidgets.QLabel(title or role.title(), card)
            title_label.setStyleSheet(
                f"font-size: 13px; font-weight: 700; color: {theme['highlight']};"
                f"background: rgba(255,255,255,0.05); border: 1px solid {theme['edge']}; border-radius: 11px; padding: 4px 10px;"
            )
            title_label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Fixed)
            body_label = QtWidgets.QLabel(body, card)
            body_label.setWordWrap(True)
            body_label.setStyleSheet(f"font-size: 13px; color: {theme['text']};")
            layout.addWidget(title_label)
            layout.addWidget(body_label)
            if chips:
                chips_label = QtWidgets.QLabel("  |  ".join(chips[:4]), card)
                chips_label.setStyleSheet(
                    f"font-size: 11px; color: {theme['muted']};"
                    f"background: rgba(255,255,255,0.04); border-radius: 10px; padding: 4px 8px;"
                )
                layout.addWidget(chips_label)
            if role == "assistant":
                row.addWidget(card)
                row.addStretch(1)
            elif role == "user":
                row.addStretch(1)
                row.addWidget(card)
            else:
                row.addStretch(1)
                row.addWidget(card)
                row.addStretch(1)
            return wrapper

        def apply_shell_state(self, shell_state: ShellState) -> None:
            started_at = time.perf_counter()
            self._shell_state = shell_state
            self._orb.set_shell_state(shell_state)
            self._refresh_window_chrome()
            self._status.setText(shell_state.status_text)
            self._sub_status.setText(shell_state.sub_status_text or shell_state.current_task_summary)
            self._input.setPlaceholderText(
                "Describe the coding task..."
                if shell_state.workspace_mode == "coding_workspace"
                else "Type a message..."
            )
            hero_metrics = "   |   ".join(shell_state.hero_metric_strip)
            self._hero_metrics.setText(hero_metrics)
            self._hero_metrics.setVisible(bool(hero_metrics))
            route_bits = []
            if shell_state.current_project:
                route_bits.append(f"Project {shell_state.current_project}")
            if shell_state.current_file:
                route_bits.append(f"File {shell_state.current_file}")
            if shell_state.active_route_summary:
                route_bits.append(f"Route {self._route_primary_label(shell_state.active_route_summary)}")
                if len(shell_state.active_route_summary) > 1:
                    route_bits.append(f"+{len(shell_state.active_route_summary) - 1} more")
            elif shell_state.active_model_roles:
                role_preview = ", ".join(shell_state.active_model_roles[:2])
                route_bits.append(
                    "Roles "
                    + role_preview
                    + (" +" + str(len(shell_state.active_model_roles) - 2) if len(shell_state.active_model_roles) > 2 else "")
                )
            route_summary = "   |   ".join(route_bits)
            self._route_summary.setText(route_summary)
            self._route_summary.setToolTip(route_summary)
            self._route_summary.setVisible(bool(route_summary))
            latest_notification = shell_state.shell_notifications[-1] if shell_state.shell_notifications else None
            self._notification.setText(latest_notification.message if latest_notification else "")
            latest_notification_id = latest_notification.notification_id if latest_notification is not None else ""
            if latest_notification_id and latest_notification_id != self._last_notification_id:
                self._notification_flash = 1.0
            self._last_notification_id = latest_notification_id
            self._ribbon.setVisible(bool(shell_state.panel_visibility_state.get("resource_ribbon", True)))
            self._activity_bar.setVisible(
                bool(shell_state.panel_visibility_state.get("activity_strip", True) and shell_state.activity_chips)
            )
            self._notification.setVisible(
                bool(
                    shell_state.panel_visibility_state.get("notifications", True)
                    and latest_notification
                    and str(latest_notification.message).strip()
                )
            )
            self._queue_theme_transition()
            self._advance_theme_transition()
            self._refresh_hero_surfaces()
            self._refresh_center_cards()
            self._refresh_context_surfaces()
            self._refresh_long_horizon_tray()
            self._refresh_dock_controls()
            _clear(self._activity_layout)
            self._activity_layout.addStretch(1)
            self._activity_chip_widgets = []
            theme = self._current_theme()
            visible_chip_limit = 5 if self.width() < 1320 else 7 if self.width() < 1560 else 9
            visible_chips = list(shell_state.activity_chips[:visible_chip_limit])
            overflow = len(shell_state.activity_chips) - len(visible_chips)
            if overflow > 0:
                visible_chips.append(
                    type(shell_state.activity_chips[0]).from_dict(
                        {
                            "chip_id": "overflow",
                            "label": f"+{overflow} more",
                            "tone": "muted",
                            "detail": "Additional live subsystem chips are collapsed to keep the hero readable.",
                            "active": False,
                        }
                    )
                    if shell_state.activity_chips
                    else None
                )
            for chip in [chip for chip in visible_chips if chip is not None]:
                label = QtWidgets.QLabel(chip.label, self._activity_bar)
                label.setToolTip(chip.detail)
                self._activity_layout.addWidget(label)
                self._activity_chip_widgets.append((label, bool(chip.active), chip.tone))
            self._activity_layout.addStretch(1)
            self._activity_bar.setVisible(
                bool(shell_state.panel_visibility_state.get("activity_strip", True) and self._activity_chip_widgets)
            )
            self._refresh_live_badge_styles()
            _clear(self._conversation_stream_layout)
            items = shell_state.conversation_items or ()
            previous_item_ids = self._conversation_item_ids
            previous_item_id_set = set(previous_item_ids)
            next_item_ids = tuple(item.item_id for item in items if item.item_id)
            intro_animation_count = 0
            if not items:
                self._conversation_stream_layout.addWidget(
                    self._message_card(
                        "system",
                        "Ready",
                        shell_state.current_task_summary or "The shell is ready for a new task.",
                        (),
                    )
                )
            else:
                for item in items:
                    card = self._message_card(item.role, item.title, item.body, item.chips)
                    self._conversation_stream_layout.addWidget(card)
                    if item.item_id and item.item_id not in previous_item_id_set:
                        intro_animation_count += 1
                        self._animate_widget_intro(card)
            self._conversation_stream_layout.addStretch(1)
            self._conversation_item_ids = next_item_ids
            self._record_intro_animation_batch(intro_animation_count)
            self._record_event_burst(
                len(shell_state.activity_chips)
                + len(shell_state.timeline_entries)
                + len(shell_state.conversation_items)
                + len(shell_state.shell_notifications)
            )
            self._bounded_autoscroll(force=bool(next_item_ids and next_item_ids != previous_item_ids))
            self._timeline.clear()
            timeline_entries = shell_state.timeline_entries[-24:]
            timeline_categories = [self._timeline_category(entry.stage) for entry in timeline_entries]
            for entry in timeline_entries:
                category = self._timeline_category(entry.stage)
                item = QtWidgets.QListWidgetItem(f"[{category}] {entry.label}\n{entry.detail}".strip())
                item.setForeground(
                    QtGui.QColor(
                        theme["timeline_warning"]
                        if entry.severity == "warning"
                        else theme["timeline_error"]
                        if entry.severity == "error"
                        else theme["timeline_info"]
                    )
                )
                item.setData(
                    _ITEM_DETAIL_ROLE,
                    "\n".join(
                        (
                            f"Entry: {entry.entry_id or '(generated)'}",
                            f"Category: {category}",
                            f"Stage: {entry.stage or '(none)'}",
                            f"Severity: {entry.severity}",
                            f"Detail: {entry.detail or '(none)'}",
                        )
                    ),
                )
                self._timeline.addItem(item)
            if self._timeline.count():
                self._timeline.setCurrentRow(0)
            warning_count = sum(1 for entry in timeline_entries if entry.severity == "warning")
            error_count = sum(1 for entry in timeline_entries if entry.severity == "error")
            branch_count = sum(1 for category in timeline_categories if category in {"Fallback", "Degraded", "Control"})
            milestone_count = sum(
                1 for category in timeline_categories if category in {"Reasoning", "Verification", "Checkpoint", "Response", "Coding"}
            )
            self._timeline_panel.set_summary(
                (
                    f"{len(timeline_entries)} events | {milestone_count} milestones | {branch_count} branches | {warning_count} warnings | {error_count} errors"
                    if timeline_entries
                    else "No timeline events recorded for the current task."
                )
            )
            self._timeline_panel.set_empty_detail("No timeline event selected.")
            self._refresh_drawer_surfaces()
            self._refresh_bottom_surfaces()
            self._apply_theme()
            self._record_perf_sample("shell_apply_ms", (time.perf_counter() - started_at) * 1000.0)
            self._refresh_debug_surface()

        def apply_dashboard_state(self, app_state: DashboardAppState) -> None:
            started_at = time.perf_counter()
            previous_ui = dict(self._ui_preferences)
            self._app_state = app_state
            self._ui_preferences = dict(app_state.user_settings.ui)
            self._refresh_window_chrome()
            self._slider.blockSignals(True)
            self._slider.setValue(max(1, min(720, int(app_state.user_settings.reasoning.get("thinking_minutes", 30) or 30))))
            self._slider.blockSignals(False)
            self._thinking.setText(f"Thinking: {self._slider.value()} min")
            self._animation_clock.apply_ui_preferences(app_state.user_settings.ui)
            self._orb.set_ui_preferences(app_state.user_settings.ui)
            self._queue_theme_transition()
            self._advance_theme_transition()
            self._mic.setEnabled("speech_to_text" in app_state.user_settings.models.get("enabled_roles", ()))
            self._refresh_center_cards()
            self._settings.setPlainText(
                "\n".join(
                    (
                        f"Profile: {app_state.user_settings.profile_name}",
                        f"Shell: {app_state.user_settings.ui.get('app_shell', 'tkinter')}",
                        f"Preset: {app_state.user_settings.ui.get('shell_preset', 'balanced')}",
                        f"Reduced motion: {app_state.user_settings.ui.get('reduced_motion', False)}",
                        f"Reduced effects: {app_state.user_settings.ui.get('reduced_effects_mode', False)}",
                        f"Lightweight mode: {app_state.user_settings.ui.get('lightweight_mode', False)}",
                        f"Low-resource mode: {app_state.user_settings.ui.get('low_resource_mode', False)}",
                        f"Adaptive drawers: {app_state.user_settings.ui.get('adaptive_drawers', True)}",
                        f"Animation cap: {app_state.user_settings.ui.get('animation_frame_cap', 30)}",
                        f"Orb size: {app_state.user_settings.ui.get('orb_size', 100)}",
                        f"Text scale: {app_state.user_settings.ui.get('status_text_scale', 1.0)}",
                        f"Reasoning mode: {app_state.user_settings.reasoning.get('mode', 'auto')}",
                    )
                )
            )
            self._readiness.setPlainText("\n".join((f"Stub ready: {app_state.readiness_report.stub_mode_ready}", f"Real mode ready: {app_state.readiness_report.real_mode_ready}", "", *[f"{check.title}: {check.status} | {check.detail}" for check in app_state.readiness_report.checks])) or "Readiness has not been loaded yet.")
            self._capability.setPlainText("\n".join(f"{item.capability_name}: {item.status} | {item.reason}" for item in app_state.readiness_report.capabilities) or "Capability readiness has not been loaded yet.")
            if any(
                previous_ui.get(key) != app_state.user_settings.ui.get(key)
                for key in (
                    "left_drawer_visible",
                    "right_drawer_visible",
                    "show_utility_drawer",
                    "adaptive_drawers",
                )
            ):
                self._sync_dock_visibility(reset_manual=True)
            else:
                self._sync_dock_visibility()
            self._refresh_context_surfaces()
            self._refresh_hero_surfaces()
            self._refresh_long_horizon_tray()
            self._refresh_dock_controls()
            self._refresh_drawer_surfaces()
            self._refresh_bottom_surfaces()
            health = app_state.runtime_health
            total_ram = float(health.total_ram_gb or 0.0)
            available_ram = float(health.available_ram_gb or 0.0)
            vram_total = float(health.generation_backend_vram_gb or 0.0) + float(
                health.embedding_backend_vram_gb or 0.0
            )
            ribbon_parts = [
                f"RAM {available_ram:.1f}/{total_ram:.1f} GB" if total_ram else "RAM n/a",
                f"VRAM {vram_total:.1f} GB",
                f"Heavy slots {len(health.active_heavy_roles)}/{health.heavy_slot_limit}",
            ]
            if self._shell_state.resource_pressure_level != "nominal":
                ribbon_parts.append(f"Governor {self._shell_state.resource_pressure_level}")
            if health.governor_summary:
                ribbon_parts.append(f"Governor note {health.governor_summary}")
            if self._shell_state.degraded_reason:
                ribbon_parts.append(f"Degraded {self._shell_state.degraded_reason}")
            if self._shell_state.capability_session_state not in {"", "inactive"}:
                ribbon_parts.append(f"Capability {self._shell_state.capability_session_state}")
            if self._shell_state.approval_pending:
                ribbon_parts.append("Approval pending")
            if self._shell_state.cloud_helper_state != "disabled":
                ribbon_parts.append("Cloud helper available")
            if self._shell_state.observation_tier and self._shell_state.observation_tier != "screenshot_on_demand":
                ribbon_parts.append(f"Observation {self._shell_state.observation_tier}")
            for flag in self._shell_state.resource_ribbon_flags:
                label = flag.replace("_", " ").replace(":", " ").strip().title()
                if label and label not in ribbon_parts:
                    ribbon_parts.append(label)
            self._ribbon.setText("   |   ".join(ribbon_parts))
            self._record_event_burst(
                len(app_state.task_history[:20])
                + len(app_state.knowledge_sources[:40])
                + len(app_state.readiness_report.checks)
                + len(app_state.readiness_report.capabilities)
                + len(app_state.model_registry_view.last_route_decisions[:6])
                + len(app_state.runtime_health.active_heavy_roles)
            )
            self._record_perf_sample("dashboard_apply_ms", (time.perf_counter() - started_at) * 1000.0)
            self._refresh_debug_surface()

        def _apply_theme(self) -> None:
            theme = self._current_theme()
            status_scale = float(self._ui_preferences.get("status_text_scale", 1.0) or 1.0)
            higher_contrast = bool(self._ui_preferences.get("higher_contrast", False))
            simple_accents = bool(self._ui_preferences.get("simple_accents", False))
            edge = "rgba(255, 255, 255, 0.18)" if higher_contrast else theme["edge"]
            button_bg = "rgba(255, 255, 255, 0.08)" if higher_contrast else theme["button_bg"]
            button_hover = theme["button_focus"] if higher_contrast else theme["button_hover"]
            button_focus = "rgba(255, 255, 255, 0.24)" if higher_contrast else theme["button_focus"]
            input_bg = "rgba(9, 16, 28, 0.95)" if self._shell_state.workspace_mode != "coding_workspace" else "rgba(24, 16, 8, 0.96)"
            accent_glow = _parse_css_color(theme["accent"])
            accent_glow.setAlpha(30)
            warning_glow = _parse_css_color(theme["warning"])
            warning_glow.setAlpha(24)
            hero_background = (
                "qlineargradient("
                "x1:0, y1:0, x2:1, y2:1, "
                f"stop:0 {_serialize_css_color(accent_glow)}, stop:0.18 {theme['panel_alt']}, stop:0.58 {theme['panel']}, stop:1 {_serialize_css_color(warning_glow)}"
                ")"
            )
            operations_background = (
                "qlineargradient("
                "x1:0, y1:0, x2:0.85, y2:1, "
                f"stop:0 {theme['panel']}, stop:0.52 {theme['panel_alt']}, stop:0.84 rgba(6, 12, 22, 0.88), stop:1 {_serialize_css_color(warning_glow)}"
                ")"
            )
            list_item_background = "rgba(255, 255, 255, 0.05)"
            list_item_selected = "rgba(255, 255, 255, 0.11)"
            self.setStyleSheet(
                "QMainWindow { background: transparent; }"
                "QWidget {"
                f"color: {theme['text']}; font-family: {theme['font_ui']};"
                "}"
                "QWidget#shellRoot {"
                f"background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {theme['background_top']}, stop:0.45 {theme['background_mid']}, stop:1 {theme['background_bottom']});"
                "}"
                "QFrame#windowChrome {"
                f"background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {theme['panel_alt']}, stop:0.52 {theme['panel']}, stop:1 rgba(12, 22, 36, 0.92)); border: 1px solid {edge}; border-top: 1px solid {theme['accent_rail']}; border-radius: 20px;"
                "}"
                "QFrame#heroSurface {"
                f"background: {hero_background}; border: 1px solid {edge}; border-top: 1px solid {theme['accent_rail']}; border-radius: 28px;"
                "}"
                "QFrame#operationsSurface {"
                f"background: {operations_background}; border: 1px solid {edge}; border-radius: 24px;"
                "}"
                "QFrame, QPlainTextEdit, QTabWidget::pane {"
                f"background: {theme['panel']}; border: 1px solid {edge}; border-radius: 16px;"
                "}"
                "QFrame#controlDock {"
                f"background: {theme['panel_alt']}; border: 1px solid {edge}; border-top: 1px solid {theme['warning']}; border-radius: 22px;"
                "}"
                "QWidget#activityBar {"
                f"background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 rgba(255,255,255,0.03), stop:0.5 {theme['glass']}, stop:1 rgba(255,255,255,0.03)); border: 1px solid {edge}; border-radius: 18px;"
                "}"
                "QLabel#shellNotification {"
                f"background: {theme['notification_bg']}; border: 1px solid {theme['notification_border']}; border-radius: 14px; padding: 8px 14px;"
                "}"
                "QSplitter::handle {"
                f"background: {theme['glass']}; border: 1px solid {edge}; border-radius: 4px; margin: 4px 0;"
                "}"
                "QDockWidget {"
                "background: transparent; border: 0px;"
                "}"
                "QDockWidget::title {"
                f"background: rgba(0,0,0,0); color: {theme['highlight']}; padding: 12px 16px 10px 16px; text-align: left; border-bottom: 1px solid {edge}; font-size: 15px; font-weight: 700;"
                "}"
                "QLineEdit {"
                f"background: {input_bg};"
                f"color: {theme['text']}; border-radius: 18px; border: 1px solid {edge}; padding: 14px 18px;"
                "}"
                "QComboBox {"
                f"background: rgba(255,255,255,0.05); color: {theme['text']}; border-radius: 13px; border: 1px solid {edge}; padding: 8px 12px;"
                "}"
                "QComboBox QAbstractItemView {"
                f"background: {theme['panel']}; color: {theme['text']}; border: 1px solid {edge}; selection-background-color: {theme['panel_alt']}; selection-color: {theme['highlight']};"
                "}"
                "QPushButton {"
                f"background: {button_bg};"
                f"color: {theme['text']}; border-radius: 14px; border: 1px solid {edge}; padding: 10px 14px;"
                "}"
                f"QPushButton:hover {{ background: {button_hover}; }}"
                f"QPushButton:focus {{ border: 1px solid {button_focus}; }}"
                f"QPushButton:checked {{ background: {theme['accent']}; color: {theme['bg']}; }}"
                "QSlider::groove:horizontal { height: 6px; background: rgba(255,255,255,0.10); border-radius: 3px; }"
                f"QSlider::handle:horizontal {{ background: {theme['accent']}; width: 16px; margin: -6px 0; border-radius: 8px; }}"
                f"QTabBar::tab {{ background: {theme['glass']}; color: {theme['muted']}; padding: 9px 14px; border-radius: 12px; margin-right: 6px; border: 1px solid {edge}; }}"
                f"QTabBar::tab:selected {{ background: {theme['chip_bg_active']}; color: {theme['text']}; border: 1px solid {theme['accent_rail']}; }}"
                f"QListWidget {{ background: transparent; border: none; padding: 2px; outline: 0px; }}"
                f"QListWidget::item {{ background: {list_item_background}; border: 1px solid {edge}; border-radius: 12px; padding: 10px 12px; margin: 4px 0px; }}"
                f"QListWidget::item:selected {{ background: {list_item_selected}; border: 1px solid {theme['accent_rail']}; color: {theme['highlight']}; }}"
                "QPlainTextEdit {"
                f"font-family: {theme['font_mono']};"
                "}"
                "QDockWidget#leftOperatorDock, QDockWidget#rightOperatorDock, QDockWidget#bottomOperatorDock {"
                "background: transparent;"
                "}"
            )
            chrome_badge_style = (
                f"padding: 6px 12px; border-radius: 12px; background: rgba(255,255,255,0.04); border: 1px solid {edge};"
            )
            chrome_signal_style = (
                f"padding: 6px 12px; border-radius: 12px; background: rgba(255,255,255,0.05); border: 1px solid {theme['accent_rail']};"
            )
            self._chrome_app_badge.setStyleSheet(
                f"padding: 6px 12px; border-radius: 12px; background: rgba(255,255,255,0.08); border: 1px solid {theme['accent_rail']};"
                f"font-size: 13px; font-weight: 800; letter-spacing: 0.8px; color: {theme['highlight']};"
            )
            for widget in (self._chrome_workspace_badge, self._chrome_profile_badge):
                widget.setStyleSheet(
                    f"{chrome_badge_style} font-size: 11px; font-weight: 700; letter-spacing: 0.6px; color: {theme['muted']};"
                )
            for widget in (self._chrome_route_badge, self._chrome_status_badge, self._chrome_signal_badge):
                widget.setStyleSheet(
                    f"{chrome_signal_style} font-size: 11px; font-weight: 700; letter-spacing: 0.6px; color: {theme['text']};"
                )
            status_color = theme["text"] if simple_accents else theme["highlight"]
            self._status.setStyleSheet(
                f"font-family: {theme['font_status']}; font-size: {int(44 * status_scale)}px; font-weight: 700; color: {status_color}; letter-spacing: 0.8px;"
            )
            info_plate = f"padding: 9px 16px; border-radius: 16px; background: rgba(255,255,255,0.03); border: 1px solid {edge};"
            self._sub_status.setStyleSheet(f"{info_plate} font-size: 15px; color: {theme['muted']};")
            self._hero_metrics.setStyleSheet(f"{info_plate} font-size: 13px; font-weight: 700; color: {theme['highlight']};")
            self._route_summary.setStyleSheet(f"{info_plate} font-size: 12px; color: {theme['muted']};")
            self._hero_agents.setStyleSheet(f"{info_plate} font-size: 12px; color: {theme['muted']};")
            self._notification.setStyleSheet(
                f"font-size: 13px; color: {theme['warning'] if self._shell_state.approval_pending or self._shell_state.degraded_reason else theme['text']};"
            )
            self._ribbon.setStyleSheet(
                f"padding: 10px 18px; border-radius: 17px; background: rgba(255,255,255,0.04); border: 1px solid {edge}; color: {theme['muted']}; font-size: 13px;"
            )
            card_border = f"1px solid {edge}"
            active_rail = (
                theme["warning"]
                if self._shell_state.fallback_reason or self._shell_state.degraded_reason or self._shell_state.approval_pending
                else theme["accent_rail"]
            )
            coding_rail = (
                theme["warning"]
                if self._shell_state.quality_gate_state not in {"", "idle", "passed"}
                else theme["accent_rail"]
            )
            final_rail = (
                theme["success"]
                if str(self._final_answer_status.text()).strip().lower() in {"valid", "passed", "completed"}
                else theme["warning"]
                if self._shell_state.degraded_reason or self._shell_state.fallback_reason
                else theme["accent_rail"]
            )
            self._approval_overlay.setStyleSheet(
                f"QFrame#approvalOverlay {{ background: {theme['card_warning']}; border: {card_border}; border-top: 2px solid {theme['warning']}; border-radius: 18px; }}"
            )
            self._long_horizon_tray.setStyleSheet(
                f"QFrame#longHorizonTray {{ background: {theme['panel_alt']}; border: {card_border}; border-top: 2px solid {theme['accent_rail']}; border-radius: 18px; }}"
            )
            self._active_task_card.setStyleSheet(
                f"QFrame#activeTaskCard {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 {theme['card_active']}, stop:1 rgba(8, 16, 28, 0.92)); border: {card_border}; border-top: 2px solid {active_rail}; border-radius: 22px; }}"
            )
            self._coding_workspace_card.setStyleSheet(
                f"QFrame#codingWorkspaceCard {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 {theme['card_coding']}, stop:1 rgba(18, 10, 5, 0.94)); border: {card_border}; border-top: 2px solid {coding_rail}; border-radius: 22px; }}"
            )
            self._final_answer_card.setStyleSheet(
                f"QFrame#finalAnswerCard {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 {theme['card_final']}, stop:1 rgba(10, 20, 32, 0.94)); border: {card_border}; border-top: 2px solid {final_rail}; border-radius: 22px; }}"
            )
            self._conversation_panel.setStyleSheet(
                f"QFrame#conversationArchivePanel {{ background: rgba(10, 22, 38, 0.62); border: {card_border}; border-radius: 22px; }}"
            )
            section_kicker_style = (
                f"padding: 3px 9px; border-radius: 10px; background: rgba(255,255,255,0.08); border: 1px solid {theme['accent_rail']};"
                f"font-size: 10px; font-weight: 800; letter-spacing: 0.9px; color: {theme['accent']};"
            )
            for widget in (
                self._approval_kicker,
                self._long_horizon_kicker,
                self._active_task_kicker,
                self._coding_workspace_kicker,
                self._final_answer_kicker,
                self._conversation_kicker,
            ):
                widget.setStyleSheet(section_kicker_style)
            self._approval_title.setStyleSheet(f"font-size: 18px; font-weight: 700; color: {theme['highlight']};")
            self._approval_risk.setStyleSheet(f"font-size: 12px; font-weight: 700; color: {theme['warning']};")
            for widget in (self._approval_summary, self._approval_target, self._approval_reason):
                widget.setStyleSheet(f"font-size: 13px; color: {theme['text']};")
            self._long_horizon_title.setStyleSheet(f"font-size: 18px; font-weight: 700; color: {theme['highlight']};")
            self._long_horizon_phase.setStyleSheet(f"font-size: 12px; font-weight: 600; color: {theme['accent']};")
            self._long_horizon_summary.setStyleSheet(f"font-size: 13px; color: {theme['text']};")
            self._long_horizon_metrics.setStyleSheet(f"font-size: 12px; color: {theme['muted']};")
            self._long_horizon_delta.setStyleSheet(f"font-size: 12px; color: {theme['highlight']};")
            title_plate = f"padding: 4px 10px; background: rgba(255,255,255,0.04); border: 1px solid {edge}; border-radius: 11px;"
            self._active_task_title.setStyleSheet(f"{title_plate} font-size: 18px; font-weight: 700; color: {theme['highlight']};")
            self._active_task_phase.setStyleSheet(f"{title_plate} font-size: 12px; font-weight: 700; color: {theme['accent']};")
            self._active_task_summary.setStyleSheet(f"font-size: 16px; font-weight: 600; color: {theme['text']};")
            self._active_task_metrics.setStyleSheet(f"font-size: 12px; color: {theme['muted']};")
            for block in (self._active_route_block, self._active_roles_block, self._active_warning_block):
                block.setStyleSheet(
                    f"QFrame#operatorDetailBlock {{ background: {theme['glass']}; border: {card_border}; border-radius: 14px; }}"
                )
            for title in (self._active_route_title, self._active_roles_title, self._active_warning_title):
                title.setStyleSheet(f"font-size: 11px; font-weight: 700; color: {theme['highlight']};")
            self._active_task_routes.setStyleSheet(f"font-size: 13px; color: {theme['text']};")
            self._active_task_roles.setStyleSheet(f"font-size: 13px; color: {theme['text']};")
            self._active_task_warnings.setStyleSheet(f"font-size: 13px; color: {theme['warning']};")
            self._coding_workspace_title.setStyleSheet(f"{title_plate} font-size: 18px; font-weight: 700; color: {theme['highlight']};")
            self._coding_workspace_status.setStyleSheet(f"{title_plate} font-size: 12px; font-weight: 700; color: {theme['accent']};")
            self._coding_workspace_summary.setStyleSheet(f"font-size: 14px; color: {theme['text']};")
            self._coding_workspace_context.setStyleSheet(f"font-size: 12px; color: {theme['muted']};")
            self._coding_workspace_validation.setStyleSheet(f"font-size: 12px; color: {theme['highlight']};")
            self._coding_workspace_artifacts.setStyleSheet(f"font-size: 12px; color: {theme['text']};")
            self._coding_workspace_blockers.setStyleSheet(f"font-size: 12px; color: {theme['warning']};")
            self._final_answer_title.setStyleSheet(f"{title_plate} font-size: 18px; font-weight: 700; color: {theme['highlight']};")
            self._final_answer_status.setStyleSheet(f"{title_plate} font-size: 12px; font-weight: 700; color: {theme['success']};")
            self._final_answer_body.setStyleSheet(f"font-size: 16px; font-weight: 600; color: {theme['text']};")
            self._final_answer_evidence.setStyleSheet(f"font-size: 12px; color: {theme['muted']};")
            self._final_answer_refs.setStyleSheet(f"font-size: 12px; color: {theme['highlight']};")
            self._final_answer_warnings.setStyleSheet(f"font-size: 12px; color: {theme['warning']};")
            self._conversation_heading.setStyleSheet(f"font-size: 15px; font-weight: 700; color: {theme['highlight']};")
            self._conversation_hint.setStyleSheet(f"font-size: 11px; color: {theme['muted']};")
            self._thinking.setStyleSheet(f"font-size: 12px; font-weight: 700; color: {theme['highlight']};")
            self._dock_status.setStyleSheet(f"font-size: 12px; color: {theme['muted']}; background: rgba(255,255,255,0.04); border: 1px solid {edge}; border-radius: 13px; padding: 7px 10px;")
            self._final_answer_sections.setStyleSheet(
                "QToolBox::tab {"
                f"background: rgba(255,255,255,0.04); color: {theme['muted']}; border-radius: 10px; padding: 8px 12px;"
                "}"
                "QToolBox::tab:selected {"
                f"background: rgba(255,255,255,0.08); color: {theme['text']};"
                "}"
            )
            for page in (self._why_answer_text, self._how_verified_text, self._deep_mode_text):
                page.setStyleSheet(
                    f"background: rgba(9, 18, 30, 0.94); color: {theme['text']}; border: {card_border}; border-radius: 12px;"
                )
            self._session_summary.setStyleSheet(f"font-size: 13px; color: {theme['text']};")
            self._history_detail.setStyleSheet(
                f"background: rgba(9, 18, 30, 0.94); color: {theme['text']}; border: {card_border}; border-radius: 12px;"
            )
            for widget in (
                self._timeline_panel.kicker_label,
                self._evidence_panel.kicker_label,
                self._provenance_panel.kicker_label,
                self._compression_panel.kicker_label,
                self._optimizer_panel.kicker_label,
                self._control_plane_panel.kicker_label,
                self._runtime_panel.kicker_label,
                self._practice_log_panel.kicker_label,
                self._coding_patterns_panel.kicker_label,
                self._coding_memory_panel.kicker_label,
                self._coding_validation_panel.kicker_label,
                self._coding_metrics_panel.kicker_label,
            ):
                widget.setStyleSheet(section_kicker_style)
            for widget in (
                self._timeline_panel.title_label,
                self._evidence_panel.title_label,
                self._provenance_panel.title_label,
                self._compression_panel.title_label,
                self._optimizer_panel.title_label,
                self._control_plane_panel.title_label,
                self._runtime_panel.title_label,
                self._practice_log_panel.title_label,
                self._coding_patterns_panel.title_label,
                self._coding_memory_panel.title_label,
                self._coding_validation_panel.title_label,
                self._coding_metrics_panel.title_label,
            ):
                widget.setStyleSheet(f"font-size: 13px; font-weight: 700; color: {theme['highlight']};")
            for widget in (
                self._timeline_panel.summary_label,
                self._evidence_panel.summary_label,
                self._provenance_panel.summary_label,
                self._compression_panel.summary_label,
                self._optimizer_panel.summary_label,
                self._control_plane_panel.summary_label,
                self._runtime_panel.summary_label,
                self._practice_log_panel.summary_label,
                self._coding_patterns_panel.summary_label,
                self._coding_memory_panel.summary_label,
                self._coding_validation_panel.summary_label,
                self._coding_metrics_panel.summary_label,
            ):
                widget.setStyleSheet(f"font-size: 12px; color: {theme['muted']};")
            for widget in (
                self._timeline_panel.detail_view,
                self._evidence_panel.detail_view,
                self._provenance_panel.detail_view,
                self._compression_panel.detail_view,
                self._optimizer_panel.detail_view,
                self._control_plane_panel.detail_view,
                self._runtime_panel.detail_view,
                self._practice_log_panel.detail_view,
                self._coding_patterns_panel.detail_view,
                self._coding_memory_panel.detail_view,
                self._coding_validation_panel.detail_view,
                self._coding_metrics_panel.detail_view,
            ):
                widget.setStyleSheet(
                    f"background: rgba(9, 18, 30, 0.94); color: {theme['text']}; border: {card_border}; border-radius: 12px;"
                )
            for widget in (
                self._timeline_panel._splitter,
                self._evidence_panel._splitter,
                self._provenance_panel._splitter,
                self._compression_panel._splitter,
                self._optimizer_panel._splitter,
                self._control_plane_panel._splitter,
                self._runtime_panel._splitter,
                self._practice_log_panel._splitter,
                self._coding_patterns_panel._splitter,
                self._coding_memory_panel._splitter,
                self._coding_validation_panel._splitter,
                self._coding_metrics_panel._splitter,
            ):
                widget.setStyleSheet(
                    f"QSplitter::handle {{ background: {theme['glass']}; border: 1px solid {edge}; border-radius: 3px; }}"
                )

        @QtCore.Slot()
        def shutdown(self) -> None:
            if self._timer.isActive():
                self._timer.stop()
            self._animation_clock.stop()
            self._orb.shutdown()
            self.close()

        def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # pragma: no cover - exercised in UI tests
            started_at = time.perf_counter()
            self._sync_dock_visibility()
            self._update_center_splitter_layout(force=False)
            super().resizeEvent(event)
            self._record_perf_sample("resize_ms", (time.perf_counter() - started_at) * 1000.0)

        def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # pragma: no cover - exercised in UI tests
            if self._timer.isActive():
                self._timer.stop()
            self._animation_clock.stop()
            self._orb.shutdown()
            super().closeEvent(event)


    class PySideShellHost:
        """Threaded host that runs the PySide shell next to the asyncio backend."""

        def __init__(
            self,
            *,
            shell_state_provider: Callable[[], ShellState],
            app_state_provider: Callable[[], DashboardAppState],
            submit_task: Callable[[str, int], bool] | None = None,
            save_settings: Callable[[UserSettingsProfile], bool] | None = None,
            request_action: Callable[[str, dict[str, Any] | None], bool] | None = None,
            startup_timeout_s: float = 10.0,
        ) -> None:
            self._shell_state_provider = shell_state_provider
            self._app_state_provider = app_state_provider
            self._submit_task = submit_task
            self._save_settings = save_settings
            self._request_action = request_action
            self._startup_timeout_s = startup_timeout_s
            self._thread: threading.Thread | None = None
            self._ready = threading.Event()
            self._stopped = threading.Event()
            self._startup_error: BaseException | None = None
            self._app: QtWidgets.QApplication | None = None
            self._window: PySideShellWindow | None = None
            self._attached_mode = False

        @property
        def is_running(self) -> bool:
            if self._attached_mode:
                return self._window is not None and self._window.isVisible()
            return bool(self._thread and self._thread.is_alive() and not self._stopped.is_set())

        def start(self) -> None:
            if self.is_running:
                return
            existing_app = QtWidgets.QApplication.instance()
            if existing_app is not None:
                self._attached_mode = True
                self._app = existing_app
                self._window = PySideShellWindow(
                    shell_state_provider=self._shell_state_provider,
                    app_state_provider=self._app_state_provider,
                    submit_task=self._submit_task,
                    save_settings=self._save_settings,
                    request_action=self._request_action,
                )
                self._window.show()
                existing_app.processEvents()
                return
            self._ready.clear()
            self._stopped.clear()
            self._startup_error = None
            self._attached_mode = False
            self._thread = threading.Thread(target=self._run, name="pyside-shell", daemon=True)
            self._thread.start()
            self._ready.wait(timeout=self._startup_timeout_s)
            if self._startup_error is not None:
                raise RuntimeError("Unable to start the PySide6 shell.") from self._startup_error
            if not self.is_running and self._startup_error is None:
                raise RuntimeError("Timed out before the PySide6 shell finished starting.")

        def stop(self, timeout_s: float = 5.0) -> None:
            if self._attached_mode:
                if self._window is not None:
                    self._window.shutdown()
                    self._window.deleteLater()
                if self._app is not None:
                    self._app.closeAllWindows()
                    self._app.processEvents()
                    QtCore.QCoreApplication.sendPostedEvents(None, 0)
                    self._app.processEvents()
                self._window = None
                self._app = None
                self._attached_mode = False
                return
            if self._window is not None:
                QtCore.QMetaObject.invokeMethod(self._window, "shutdown", QtCore.Qt.ConnectionType.QueuedConnection)
            if self._app is not None:
                QtCore.QMetaObject.invokeMethod(
                    self._app,
                    "closeAllWindows",
                    QtCore.Qt.ConnectionType.QueuedConnection,
                )
                QtCore.QMetaObject.invokeMethod(self._app, "quit", QtCore.Qt.ConnectionType.QueuedConnection)
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=timeout_s)
            self._stopped.wait(timeout=timeout_s)
            self._thread = None
            self._window = None
            self._app = None

        def refresh_from_state(self) -> bool:
            if self._window is None:
                return False
            if self._attached_mode:
                self._window.refresh_from_state()
                if self._app is not None:
                    self._app.processEvents()
                return True
            QtCore.QMetaObject.invokeMethod(
                self._window,
                "refresh_from_state",
                QtCore.Qt.ConnectionType.QueuedConnection,
            )
            return True

        def open_surface(self, surface_id: str) -> bool:
            normalized = str(surface_id).strip().lower()
            if not normalized or self._window is None:
                return False
            if self._attached_mode:
                opened = bool(self._window.open_surface(normalized))
                if opened and self._app is not None:
                    self._app.processEvents()
                return opened
            QtCore.QMetaObject.invokeMethod(
                self._window,
                "_open_surface_slot",
                QtCore.Qt.ConnectionType.QueuedConnection,
                QtCore.Q_ARG(str, normalized),
            )
            return True

        def _run(self) -> None:  # pragma: no cover - exercised through host smoke coverage
            try:
                os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
                app = QtWidgets.QApplication([])
                app.setQuitOnLastWindowClosed(True)
                self._app = app
                self._window = PySideShellWindow(
                    shell_state_provider=self._shell_state_provider,
                    app_state_provider=self._app_state_provider,
                    submit_task=self._submit_task,
                    save_settings=self._save_settings,
                    request_action=self._request_action,
                )
                self._window.show()
                self._ready.set()
                app.exec()
            except BaseException as exc:
                self._startup_error = exc
                self._ready.set()
            finally:
                self._app = None
                self._window = None
                self._stopped.set()


    def preview_shell(
        shell_state: ShellState | None = None,
        *,
        app_state: DashboardAppState | None = None,
    ) -> int:
        """Launch a local preview of the PySide shell."""
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        window = PySideShellWindow(shell_state=shell_state, app_state=app_state)
        window.show()
        return app.exec()

else:

    class OrbWidget:  # pragma: no cover - dependency-gated path
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs
            raise PySideShellUnavailableError(
                "PySide6 is not installed. Install the desktop extra to use the premium shell scaffold."
            )


    class PySideShellWindow:  # pragma: no cover - dependency-gated path
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs
            raise PySideShellUnavailableError(
                "PySide6 is not installed. Install the desktop extra to use the premium shell scaffold."
            )


    class PySideShellHost:  # pragma: no cover - dependency-gated path
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs
            raise PySideShellUnavailableError(
                "PySide6 is not installed. Install the desktop extra to use the premium shell scaffold."
            )


    def preview_shell(
        shell_state: ShellState | None = None,
        *,
        app_state: DashboardAppState | None = None,
    ) -> int:  # pragma: no cover - dependency-gated path
        del shell_state, app_state
        raise PySideShellUnavailableError(
            "PySide6 is not installed. Install the desktop extra to use the premium shell scaffold."
        )
