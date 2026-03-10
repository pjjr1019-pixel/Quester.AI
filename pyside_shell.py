"""Premium PySide6 shell host for the local desktop UI migration."""

from __future__ import annotations

import math
import os
import threading
from collections.abc import Callable
from dataclasses import replace
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

    def _theme(shell_state: ShellState) -> dict[str, str]:
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
        accent, bg, highlight = palette_map.get(shell_state.orb_palette, palette_map["calm_blue"])
        return {
            "accent": accent,
            "bg": bg,
            "highlight": highlight,
            "panel": "rgba(11, 18, 31, 0.84)",
            "edge": "rgba(255, 255, 255, 0.08)",
            "text": "#eef5ff",
            "muted": "#b6c2d7",
            "warning": "#f6c36b",
            "danger": "#ff7e8f",
            "success": "#8fe0b3",
        }

    def _clear(layout: QtWidgets.QLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            child_layout = item.layout()
            if widget is not None:
                widget.deleteLater()
            elif child_layout is not None:
                _clear(child_layout)

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
            self._particle_density = "balanced"
            self._phase = 0.0
            self._timer = QtCore.QTimer(self)
            self._timer.timeout.connect(self._on_tick)
            self._timer.start(33)
            self.setMinimumHeight(320)
            self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        def set_shell_state(self, shell_state: ShellState) -> None:
            self._shell_state = shell_state
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
            frame_cap = int(ui.get("animation_frame_cap", 30) or 30)
            interval_ms = max(8, int(1000 / max(10, min(120, frame_cap))))
            if self._reduced_motion:
                interval_ms = max(interval_ms, 66)
            elif self._minimal:
                interval_ms = max(interval_ms, 50)
            self._timer.start(interval_ms)

        def _on_tick(self) -> None:
            delta = 0.02 if self._reduced_motion else 0.03 if self._reduced_effects else 0.04
            self._phase = (self._phase + (delta * self._animation_intensity)) % (math.tau * 8.0)
            self.update()

        def shutdown(self) -> None:
            if self._timer.isActive():
                self._timer.stop()

        def _palette(self) -> tuple[QtGui.QColor, QtGui.QColor, QtGui.QColor]:
            theme = _theme(self._shell_state)
            return QtGui.QColor(theme["accent"]), QtGui.QColor(theme["bg"]), QtGui.QColor(theme["highlight"])

        def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # pragma: no cover - dependency-gated path
            del event
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.Antialiasing)
            painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)

            rect = self.rect()
            theme = _theme(self._shell_state)
            painter.fillRect(rect, QtGui.QColor(theme["bg"]))

            primary, shadow, highlight = self._palette()
            intensity = max(0.1, min(1.0, self._shell_state.orb_intensity * self._animation_intensity))
            breathe = 1.0 + (math.sin(self._phase) * 0.04 * intensity)
            center = QtCore.QPointF(rect.center().x(), rect.center().y() - (rect.height() * 0.08))
            radius = min(rect.width(), rect.height()) * 0.21 * breathe * self._orb_scale

            if self._ambient_reactivity and not self._reduced_effects:
                ambient = QtGui.QRadialGradient(center, radius * 1.9)
                ambient.setColorAt(0.0, QtGui.QColor(primary.red(), primary.green(), primary.blue(), int(72 * intensity)))
                ambient.setColorAt(0.65, QtGui.QColor(primary.red(), primary.green(), primary.blue(), int(24 * intensity)))
                ambient.setColorAt(1.0, QtGui.QColor(0, 0, 0, 0))
                painter.setPen(QtCore.Qt.PenStyle.NoPen)
                painter.setBrush(ambient)
                painter.drawEllipse(center, radius * 1.95, radius * 1.65)

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

            if not self._minimal and not self._simple_orb:
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

            if self._shell_state.orb_effects.approval_hold:
                hold_pen = QtGui.QPen(QtGui.QColor(theme["warning"]))
                hold_pen.setWidthF(max(2.0, radius * 0.02))
                painter.setPen(hold_pen)
                painter.drawEllipse(center, radius * 1.28, radius * 1.28)

            if self._shell_state.orb_effects.verification_lock_pending:
                lock_pen = QtGui.QPen(QtGui.QColor(theme["success"]))
                lock_pen.setWidthF(max(1.0, radius * 0.013))
                painter.setPen(lock_pen)
                painter.drawEllipse(center, radius * 1.45, radius * 1.45)

            if self._shell_state.orb_effects.checkpoint_pulse_pending:
                checkpoint_pen = QtGui.QPen(QtGui.QColor(theme["warning"]))
                checkpoint_pen.setWidthF(max(1.8, radius * 0.018))
                painter.setPen(checkpoint_pen)
                painter.drawEllipse(center, radius * 1.62, radius * 1.62)

            if not self._reduced_effects:
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
                    amplitude = radius * (0.09 if self._shell_state.orb_mode == "speaking" else 0.03)
                    y = waveform_y + (wave * amplitude)
                    path.lineTo(x, y)
                painter.drawPath(path)


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
            self._selected_coding_role = ""
            self._suppress_coding_route_signals = False
            self.setWindowTitle("Quester.AI")
            self.resize(1460, 980)
            self.setMinimumSize(1180, 760)
            self._build_ui()
            self._timer = QtCore.QTimer(self)
            self._timer.timeout.connect(self._poll_state)
            self._timer.start(140)
            self.apply_dashboard_state(self._app_state)
            self.apply_shell_state(self._shell_state)

        def _build_ui(self) -> None:
            central = QtWidgets.QWidget(self)
            root = QtWidgets.QVBoxLayout(central)
            root.setContentsMargins(20, 16, 20, 18)
            root.setSpacing(14)
            self._ribbon = QtWidgets.QLabel(central, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
            root.addWidget(self._ribbon)
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
            self._activity_layout = QtWidgets.QHBoxLayout(self._activity_bar)
            self._activity_layout.setContentsMargins(0, 0, 0, 0)
            self._activity_layout.setSpacing(8)
            self._activity_layout.addStretch(1)
            self._notification = QtWidgets.QLabel(alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
            self._notification.setWordWrap(True)
            root.addWidget(self._orb, stretch=5)
            root.addWidget(self._status)
            root.addWidget(self._sub_status)
            root.addWidget(self._hero_metrics)
            root.addWidget(self._route_summary)
            root.addWidget(self._hero_agents)
            root.addWidget(self._activity_bar)
            root.addWidget(self._notification)
            self._policy_context_bar = QtWidgets.QWidget(central)
            self._policy_context_layout = QtWidgets.QHBoxLayout(self._policy_context_bar)
            self._policy_context_layout.setContentsMargins(0, 0, 0, 0)
            self._policy_context_layout.setSpacing(8)
            self._policy_context_layout.addStretch(1)
            root.addWidget(self._policy_context_bar)
            self._approval_overlay = QtWidgets.QFrame(central)
            self._approval_overlay.setObjectName("approvalOverlay")
            approval_layout = QtWidgets.QVBoxLayout(self._approval_overlay)
            approval_layout.setContentsMargins(18, 16, 18, 16)
            approval_layout.setSpacing(8)
            approval_header = QtWidgets.QHBoxLayout()
            self._approval_title = QtWidgets.QLabel("Approval Required", self._approval_overlay)
            self._approval_risk = QtWidgets.QLabel("", self._approval_overlay)
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
            root.addWidget(self._approval_overlay)
            self._long_horizon_tray = QtWidgets.QFrame(central)
            self._long_horizon_tray.setObjectName("longHorizonTray")
            long_horizon_layout = QtWidgets.QVBoxLayout(self._long_horizon_tray)
            long_horizon_layout.setContentsMargins(18, 14, 18, 14)
            long_horizon_layout.setSpacing(8)
            long_horizon_header = QtWidgets.QHBoxLayout()
            self._long_horizon_title = QtWidgets.QLabel("Long-Horizon", self._long_horizon_tray)
            self._long_horizon_phase = QtWidgets.QLabel("", self._long_horizon_tray)
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
            root.addWidget(self._long_horizon_tray)
            self._conversation_scroll = QtWidgets.QScrollArea(central)
            self._conversation_scroll.setWidgetResizable(True)
            self._conversation_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
            self._conversation_container = QtWidgets.QWidget(self._conversation_scroll)
            self._conversation_layout = QtWidgets.QVBoxLayout(self._conversation_container)
            self._conversation_layout.setContentsMargins(0, 0, 0, 0)
            self._conversation_layout.setSpacing(12)
            self._active_task_card = QtWidgets.QFrame(self._conversation_container)
            self._active_task_card.setObjectName("activeTaskCard")
            active_layout = QtWidgets.QVBoxLayout(self._active_task_card)
            active_layout.setContentsMargins(18, 16, 18, 16)
            active_layout.setSpacing(8)
            active_header = QtWidgets.QHBoxLayout()
            self._active_task_title = QtWidgets.QLabel("Active Task", self._active_task_card)
            self._active_task_phase = QtWidgets.QLabel("", self._active_task_card)
            active_header.addWidget(self._active_task_title)
            active_header.addStretch(1)
            active_header.addWidget(self._active_task_phase)
            active_layout.addLayout(active_header)
            self._active_task_summary = QtWidgets.QLabel(self._active_task_card)
            self._active_task_summary.setWordWrap(True)
            self._active_task_summary.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            active_layout.addWidget(self._active_task_summary)
            self._active_task_metrics = QtWidgets.QLabel(self._active_task_card)
            self._active_task_metrics.setWordWrap(True)
            self._active_task_metrics.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            active_layout.addWidget(self._active_task_metrics)
            self._active_task_routes = QtWidgets.QLabel(self._active_task_card)
            self._active_task_routes.setWordWrap(True)
            self._active_task_routes.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            active_layout.addWidget(self._active_task_routes)
            self._active_task_warnings = QtWidgets.QLabel(self._active_task_card)
            self._active_task_warnings.setWordWrap(True)
            self._active_task_warnings.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            active_layout.addWidget(self._active_task_warnings)
            self._conversation_layout.addWidget(self._active_task_card)

            self._coding_workspace_card = QtWidgets.QFrame(self._conversation_container)
            self._coding_workspace_card.setObjectName("codingWorkspaceCard")
            coding_layout = QtWidgets.QVBoxLayout(self._coding_workspace_card)
            coding_layout.setContentsMargins(18, 16, 18, 16)
            coding_layout.setSpacing(8)
            coding_header = QtWidgets.QHBoxLayout()
            self._coding_workspace_title = QtWidgets.QLabel("Coding Workspace", self._coding_workspace_card)
            self._coding_workspace_status = QtWidgets.QLabel("", self._coding_workspace_card)
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
            self._conversation_layout.addWidget(self._coding_workspace_card)

            self._final_answer_card = QtWidgets.QFrame(self._conversation_container)
            self._final_answer_card.setObjectName("finalAnswerCard")
            final_layout = QtWidgets.QVBoxLayout(self._final_answer_card)
            final_layout.setContentsMargins(18, 16, 18, 16)
            final_layout.setSpacing(8)
            final_header = QtWidgets.QHBoxLayout()
            self._final_answer_title = QtWidgets.QLabel("Final Answer", self._final_answer_card)
            self._final_answer_status = QtWidgets.QLabel("", self._final_answer_card)
            final_header.addWidget(self._final_answer_title)
            final_header.addStretch(1)
            final_header.addWidget(self._final_answer_status)
            final_layout.addLayout(final_header)
            self._final_answer_body = QtWidgets.QLabel(self._final_answer_card)
            self._final_answer_body.setWordWrap(True)
            self._final_answer_body.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
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
            self._why_answer_text = QtWidgets.QPlainTextEdit(self._final_answer_sections)
            self._why_answer_text.setReadOnly(True)
            self._how_verified_text = QtWidgets.QPlainTextEdit(self._final_answer_sections)
            self._how_verified_text.setReadOnly(True)
            self._deep_mode_text = QtWidgets.QPlainTextEdit(self._final_answer_sections)
            self._deep_mode_text.setReadOnly(True)
            self._final_answer_sections.addItem(self._why_answer_text, "Why This Output")
            self._final_answer_sections.addItem(self._how_verified_text, "How It Was Verified")
            self._final_answer_sections.addItem(self._deep_mode_text, "What Deep Mode Changed")
            final_layout.addWidget(self._final_answer_sections)
            final_actions = QtWidgets.QHBoxLayout()
            self._copy_answer_button = QtWidgets.QPushButton("Copy Answer", self._final_answer_card)
            self._copy_citations_button = QtWidgets.QPushButton("Copy Citations", self._final_answer_card)
            self._copy_answer_button.clicked.connect(self._copy_final_answer)
            self._copy_citations_button.clicked.connect(self._copy_final_references)
            final_actions.addWidget(self._copy_answer_button)
            final_actions.addWidget(self._copy_citations_button)
            final_actions.addStretch(1)
            final_layout.addLayout(final_actions)
            self._conversation_layout.addWidget(self._final_answer_card)

            self._conversation_stream = QtWidgets.QWidget(self._conversation_container)
            self._conversation_stream_layout = QtWidgets.QVBoxLayout(self._conversation_stream)
            self._conversation_stream_layout.setContentsMargins(0, 0, 0, 0)
            self._conversation_stream_layout.setSpacing(12)
            self._conversation_layout.addWidget(self._conversation_stream)
            self._conversation_scroll.setWidget(self._conversation_container)
            root.addWidget(self._conversation_scroll, stretch=4)
            dock = QtWidgets.QFrame(central)
            dock_layout = QtWidgets.QVBoxLayout(dock)
            dock_layout.setContentsMargins(16, 14, 16, 14)
            dock_layout.setSpacing(10)
            top = QtWidgets.QHBoxLayout()
            self._mic = QtWidgets.QPushButton("Mic", dock)
            self._mic.setEnabled(False)
            self._input = QtWidgets.QLineEdit(dock)
            self._input.setPlaceholderText("Type a message...")
            self._input.returnPressed.connect(self._submit_from_input)
            self._send = QtWidgets.QPushButton("Send", dock)
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
            self._history_shortcut = QtWidgets.QPushButton("History", dock)
            self._history_shortcut.clicked.connect(lambda: self._show_bottom_tab(0))
            self._settings_shortcut = QtWidgets.QPushButton("Settings", dock)
            self._settings_shortcut.clicked.connect(lambda: self._show_bottom_tab(2))
            self._readiness_shortcut = QtWidgets.QPushButton("Readiness", dock)
            self._readiness_shortcut.clicked.connect(self._refresh_readiness_surfaces)
            self._support_bundle_button = QtWidgets.QPushButton("Support Bundle", dock)
            self._support_bundle_button.clicked.connect(self._export_support_bundle)
            self._trace_export_button = QtWidgets.QPushButton("Export Trace", dock)
            self._trace_export_button.clicked.connect(self._export_selected_trace)
            for widget in (
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
            root.addWidget(dock)
            self.setCentralWidget(central)

            self._agent_list = QtWidgets.QListWidget()
            self._agent_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
            self._timeline = QtWidgets.QListWidget()
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
            self._left_tabs.addTab(self._agent_list, "Agents")
            self._left_tabs.addTab(self._timeline, "Timeline")
            self._left_tabs.addTab(self._session_panel, "Session")
            self._left_dock = QtWidgets.QDockWidget("Task Timeline", self)
            self._left_dock.setWidget(self._left_tabs)
            self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, self._left_dock)

            self._evidence_list = QtWidgets.QListWidget()
            self._provenance_list = QtWidgets.QListWidget()
            self._compression_list = QtWidgets.QListWidget()
            self._optimizer_list = QtWidgets.QListWidget()
            self._control_plane_list = QtWidgets.QListWidget()
            self._runtime_list = QtWidgets.QListWidget()
            self._practice_log_list = QtWidgets.QListWidget()
            self._coding_patterns_list = QtWidgets.QListWidget()
            self._coding_validation_list = QtWidgets.QListWidget()
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
            self._right_tabs.addTab(self._evidence_list, "Evidence")
            self._right_tabs.addTab(self._provenance_list, "Provenance")
            self._right_tabs.addTab(self._compression_list, "Compressor")
            self._right_tabs.addTab(self._optimizer_list, "Optimizer")
            self._right_tabs.addTab(self._control_plane_list, "Control Plane")
            self._right_tabs.addTab(self._runtime_list, "Runtime")
            self._right_tabs.addTab(self._practice_log_list, "Practice")
            self._right_tabs.addTab(self._coding_patterns_list, "Patterns")
            self._right_tabs.addTab(self._coding_validation_list, "Validation")
            self._right_tabs.addTab(self._coding_routes_panel, "Coding Routes")
            self._right_dock = QtWidgets.QDockWidget("Evidence And Insights", self)
            self._right_dock.setWidget(self._right_tabs)
            self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self._right_dock)

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
            for name, widget in (("History", self._history_panel), ("Knowledge", self._knowledge_list), ("Settings", self._settings), ("Readiness", self._readiness), ("Capabilities", self._capability), ("Debug", self._debug)):
                self._bottom_tabs.addTab(widget, name)
            self._bottom_dock = QtWidgets.QDockWidget("Secondary Views", self)
            self._bottom_dock.setWidget(self._bottom_tabs)
            self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, self._bottom_dock)
            initial_ui = self._app_state.user_settings.ui
            self._left_dock.setVisible(bool(initial_ui.get("left_drawer_visible", True)))
            self._right_dock.setVisible(bool(initial_ui.get("right_drawer_visible", True)))
            self._bottom_dock.setVisible(bool(initial_ui.get("show_utility_drawer", False)))

        def _poll_state(self) -> None:
            if self._shell_state_provider is not None:
                self.apply_shell_state(self._shell_state_provider())
            if self._app_state_provider is not None:
                self.apply_dashboard_state(self._app_state_provider())

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

        def _show_bottom_tab(self, index: int) -> None:
            self._bottom_dock.show()
            self._bottom_tabs.setCurrentIndex(max(0, min(self._bottom_tabs.count() - 1, int(index))))

        def _default_export_path(self, stem: str, suffix: str) -> str:
            safe_profile = (
                str(self._app_state.user_settings.profile_name or "default")
                .strip()
                .replace(" ", "_")
                .replace("/", "_")
            )
            return str(Path("logs") / "shell_exports" / f"{safe_profile}_{stem}{suffix}")

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

        def _tone_color(self, tone: str) -> str:
            theme = _theme(self._shell_state)
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
            theme = _theme(self._shell_state)
            label.setStyleSheet(
                "QLabel {"
                f"color: {color}; background: rgba(255,255,255,0.04);"
                f"border: 1px solid {theme['edge']}; border-radius: 12px; padding: 6px 10px;"
                "font-size: 12px; font-weight: 600;"
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
        ) -> None:
            current_value = ""
            current_item = widget.currentItem()
            if current_item is not None:
                current_value = str(current_item.data(QtCore.Qt.ItemDataRole.UserRole) or "")
            widget.clear()
            theme = _theme(self._shell_state)
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
            dock_summary = [
                f"Mode {mode_label}",
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
            self._populate_list(self._evidence_list, evidence_entries)

            provenance_entries = [
                ("Task id", task.task_id or "(none)", "info"),
                ("Stage", task.running_stage or app_state.last_stage or "(idle)", "accent"),
                ("Verifier", task.selected_verifier or "(none)", "accent"),
                ("Strategy", task.selected_strategy or "(none)", "muted"),
                ("Candidate score", f"{task.candidate_score:.2f}", "success" if task.candidate_score >= 0.75 else "warning"),
                ("Repairs", ", ".join(task.repair_actions) if task.repair_actions else "(none)", "info"),
                ("Failures", ", ".join(task.failure_categories) if task.failure_categories else "(none)", "danger" if task.failure_categories else "muted"),
            ]
            self._populate_list(self._provenance_list, provenance_entries)

            compression_entries = [
                (
                    insight.macro_name or insight.proposal_id,
                    f"gain {insight.estimated_gain:.2f} | pass {insight.validation_pass_rate:.2f} | {insight.validation_state}",
                    "success" if insight.accepted else "warning",
                )
                for insight in registry.compression_insights[:8]
            ] or [("Compression", shell_state.compression_state or "idle", "muted")]
            self._populate_list(self._compression_list, compression_entries)

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
            self._populate_list(self._optimizer_list, optimizer_entries)

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
            self._populate_list(self._control_plane_list, control_entries)

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
            self._populate_list(self._runtime_list, runtime_entries)

            practice_entries = [
                (
                    coding_practice.prompt or coding_practice.session_id or "Coding Dojo",
                    (
                        f"{coding_practice.status} | score {coding_practice.quality_score:.2f} | "
                        f"validated {len(coding_practice.validated_patterns)} | rejected {len(coding_practice.rejected_patterns)}"
                    ),
                    "accent" if coding_practice.status not in {"", "idle"} else "muted",
                )
            ]
            if coding_practice.warnings:
                practice_entries.extend(
                    ("Practice warning", warning, "warning") for warning in coding_practice.warnings[:4]
                )
            self._populate_list(
                self._practice_log_list,
                practice_entries or [("Practice", "No coding practice log loaded.", "muted")],
            )

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
            self._populate_list(self._coding_patterns_list, pattern_entries)

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
            if coding_output.warnings:
                validation_entries.extend(
                    ("Coding warning", warning, "warning") for warning in coding_output.warnings[:4]
                )
            self._populate_list(self._coding_validation_list, validation_entries)
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
                    f"Export path: {selected_task.trace_debug_export_path or '(none)'}",
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
            self._active_task_routes.setVisible(bool(routes_text))
            self._active_task_warnings.setText("   |   ".join(warning_parts[:3]))
            self._active_task_warnings.setVisible(bool(warning_parts))

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
            artifact_lines = [
                f"{artifact.title or artifact.artifact_type}: {artifact.path or artifact.content_preview[:60]}"
                for artifact in coding_result.artifacts[:4]
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
                    for artifact in coding_result.artifacts[:4]
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

        def _message_card(self, role: str, title: str, body: str, chips: tuple[str, ...]) -> QtWidgets.QWidget:
            theme = _theme(self._shell_state)
            wrapper = QtWidgets.QWidget(self._conversation_container)
            row = QtWidgets.QHBoxLayout(wrapper)
            row.setContentsMargins(0, 0, 0, 0)
            card = QtWidgets.QFrame(wrapper)
            card.setMaximumWidth(760)
            color = "rgba(22, 40, 68, 0.86)" if role == "user" else "rgba(13, 23, 41, 0.92)" if role == "assistant" else "rgba(34, 24, 28, 0.88)"
            card.setStyleSheet(f"QFrame {{ background: {color}; border: 1px solid {theme['edge']}; border-radius: 20px; }}")
            layout = QtWidgets.QVBoxLayout(card)
            layout.setContentsMargins(16, 14, 16, 14)
            title_label = QtWidgets.QLabel(title or role.title(), card)
            title_label.setStyleSheet("font-size: 14px; font-weight: 700;")
            body_label = QtWidgets.QLabel(body, card)
            body_label.setWordWrap(True)
            layout.addWidget(title_label)
            layout.addWidget(body_label)
            if chips:
                chips_label = QtWidgets.QLabel("  |  ".join(chips[:4]), card)
                chips_label.setStyleSheet("font-size: 11px; color: #dbe8ff;")
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
            self._shell_state = shell_state
            self._orb.set_shell_state(shell_state)
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
                route_bits.append("Routes " + " | ".join(shell_state.active_route_summary[:4]))
            route_summary = "   |   ".join(route_bits)
            self._route_summary.setText(route_summary)
            self._route_summary.setToolTip(route_summary)
            self._route_summary.setVisible(bool(route_summary))
            self._notification.setText(shell_state.shell_notifications[-1].message if shell_state.shell_notifications else "")
            self._ribbon.setVisible(bool(shell_state.panel_visibility_state.get("resource_ribbon", True)))
            self._activity_bar.setVisible(bool(shell_state.panel_visibility_state.get("activity_strip", True)))
            self._notification.setVisible(bool(shell_state.panel_visibility_state.get("notifications", True)))
            self._refresh_hero_surfaces()
            self._refresh_center_cards()
            self._refresh_context_surfaces()
            self._refresh_long_horizon_tray()
            self._refresh_dock_controls()
            _clear(self._activity_layout)
            self._activity_layout.addStretch(1)
            theme = _theme(shell_state)
            for chip in shell_state.activity_chips:
                label = QtWidgets.QLabel(chip.label, self._activity_bar)
                color = theme["warning"] if chip.tone == "warning" else theme["danger"] if chip.tone == "danger" else theme["highlight"] if chip.tone == "accent" else theme["accent"]
                label.setStyleSheet(f"QLabel {{ color: {color}; background: rgba(255,255,255,0.04); border: 1px solid {theme['edge']}; border-radius: 12px; padding: 6px 12px; font-size: 12px; font-weight: 600; }}")
                label.setToolTip(chip.detail)
                self._activity_layout.addWidget(label)
            self._activity_layout.addStretch(1)
            _clear(self._conversation_stream_layout)
            items = shell_state.conversation_items or ()
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
                    self._conversation_stream_layout.addWidget(
                        self._message_card(item.role, item.title, item.body, item.chips)
                    )
            self._conversation_stream_layout.addStretch(1)
            self._timeline.clear()
            for entry in shell_state.timeline_entries[-24:]:
                item = QtWidgets.QListWidgetItem(f"{entry.label}\n{entry.detail}".strip())
                item.setForeground(QtGui.QColor(theme["warning"] if entry.severity == "warning" else theme["danger"] if entry.severity == "error" else theme["text"]))
                self._timeline.addItem(item)
            self._refresh_drawer_surfaces()
            self._refresh_bottom_surfaces()
            self._apply_theme()

        def apply_dashboard_state(self, app_state: DashboardAppState) -> None:
            previous_ui = dict(self._ui_preferences)
            self._app_state = app_state
            self._ui_preferences = dict(app_state.user_settings.ui)
            self._slider.blockSignals(True)
            self._slider.setValue(max(1, min(720, int(app_state.user_settings.reasoning.get("thinking_minutes", 30) or 30))))
            self._slider.blockSignals(False)
            self._thinking.setText(f"Thinking: {self._slider.value()} min")
            self._orb.set_ui_preferences(app_state.user_settings.ui)
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
                        f"Animation cap: {app_state.user_settings.ui.get('animation_frame_cap', 30)}",
                        f"Orb size: {app_state.user_settings.ui.get('orb_size', 100)}",
                        f"Text scale: {app_state.user_settings.ui.get('status_text_scale', 1.0)}",
                        f"Reasoning mode: {app_state.user_settings.reasoning.get('mode', 'auto')}",
                    )
                )
            )
            self._readiness.setPlainText("\n".join((f"Stub ready: {app_state.readiness_report.stub_mode_ready}", f"Real mode ready: {app_state.readiness_report.real_mode_ready}", "", *[f"{check.title}: {check.status} | {check.detail}" for check in app_state.readiness_report.checks])) or "Readiness has not been loaded yet.")
            self._capability.setPlainText("\n".join(f"{item.capability_name}: {item.status} | {item.reason}" for item in app_state.readiness_report.capabilities) or "Capability readiness has not been loaded yet.")
            self._debug.setPlainText("\n".join((f"last_stage: {app_state.last_stage}", f"event_count: {app_state.event_count}", f"dropped_events: {app_state.dropped_events}", f"last_notice: {app_state.last_notice}", f"summary: {self._shell_state.current_task_summary}")))
            if previous_ui.get("left_drawer_visible", True) != app_state.user_settings.ui.get("left_drawer_visible", True):
                self._left_dock.setVisible(bool(app_state.user_settings.ui.get("left_drawer_visible", True)))
            if previous_ui.get("right_drawer_visible", True) != app_state.user_settings.ui.get("right_drawer_visible", True):
                self._right_dock.setVisible(bool(app_state.user_settings.ui.get("right_drawer_visible", True)))
            if previous_ui.get("show_utility_drawer", False) != app_state.user_settings.ui.get("show_utility_drawer", False):
                self._bottom_dock.setVisible(bool(app_state.user_settings.ui.get("show_utility_drawer", False)))
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
            if self._shell_state.cloud_helper_state != "disabled":
                ribbon_parts.append("Cloud helper available")
            if self._shell_state.observation_tier and self._shell_state.observation_tier != "screenshot_on_demand":
                ribbon_parts.append(f"Observation {self._shell_state.observation_tier}")
            for flag in self._shell_state.resource_ribbon_flags:
                label = flag.replace("_", " ").replace(":", " ").strip().title()
                if label and label not in ribbon_parts:
                    ribbon_parts.append(label)
            self._ribbon.setText("   |   ".join(ribbon_parts))

        def _apply_theme(self) -> None:
            theme = _theme(self._shell_state)
            status_scale = float(self._ui_preferences.get("status_text_scale", 1.0) or 1.0)
            higher_contrast = bool(self._ui_preferences.get("higher_contrast", False))
            simple_accents = bool(self._ui_preferences.get("simple_accents", False))
            edge = "rgba(255, 255, 255, 0.18)" if higher_contrast else theme["edge"]
            button_bg = "rgba(255, 255, 255, 0.08)" if higher_contrast else "rgba(255, 255, 255, 0.04)"
            self.setStyleSheet(
                "QMainWindow, QWidget {"
                f"background: {theme['bg']}; color: {theme['text']};"
                "}"
                "QPlainTextEdit, QListWidget, QTabWidget::pane, QFrame {"
                f"background: {theme['panel']}; border: 1px solid {edge}; border-radius: 16px;"
                "}"
                "QLineEdit {"
                "background: rgba(10, 17, 29, 0.95);"
                f"color: {theme['text']}; border-radius: 18px; border: 1px solid {edge}; padding: 14px 18px;"
                "}"
                "QPushButton {"
                f"background: {button_bg};"
                f"color: {theme['text']}; border-radius: 14px; border: 1px solid {edge}; padding: 10px 14px;"
                "}"
                f"QPushButton:checked {{ background: {theme['accent']}; color: {theme['bg']}; }}"
                "QSlider::groove:horizontal { height: 6px; background: rgba(255,255,255,0.10); border-radius: 3px; }"
                f"QSlider::handle:horizontal {{ background: {theme['accent']}; width: 16px; margin: -6px 0; border-radius: 8px; }}"
                "QTabBar::tab { background: rgba(255,255,255,0.03); color: #b6c2d7; padding: 8px 12px; border-top-left-radius: 10px; border-top-right-radius: 10px; margin-right: 4px; }"
                "QTabBar::tab:selected { background: rgba(255,255,255,0.08); color: #eef5ff; }"
            )
            status_color = theme["text"] if simple_accents else theme["highlight"]
            self._status.setStyleSheet(
                f"font-size: {int(34 * status_scale)}px; font-weight: 700; color: {status_color};"
            )
            self._sub_status.setStyleSheet(f"font-size: 15px; color: {theme['muted']};")
            self._hero_metrics.setStyleSheet(f"font-size: 13px; font-weight: 600; color: {theme['highlight']};")
            self._route_summary.setStyleSheet(f"font-size: 12px; color: {theme['muted']};")
            self._hero_agents.setStyleSheet(f"font-size: 12px; color: {theme['muted']};")
            self._notification.setStyleSheet(f"font-size: 13px; color: {theme['warning'] if self._shell_state.approval_pending or self._shell_state.degraded_reason else theme['muted']};")
            self._ribbon.setStyleSheet(f"padding: 8px 12px; border-radius: 14px; background: rgba(255,255,255,0.03); border: 1px solid {edge}; color: {theme['muted']};")
            card_border = f"1px solid {edge}"
            self._approval_overlay.setStyleSheet(
                f"QFrame#approvalOverlay {{ background: rgba(46, 31, 14, 0.95); border: {card_border}; border-radius: 18px; }}"
            )
            self._long_horizon_tray.setStyleSheet(
                f"QFrame#longHorizonTray {{ background: rgba(22, 30, 17, 0.95); border: {card_border}; border-radius: 18px; }}"
            )
            self._active_task_card.setStyleSheet(
                f"QFrame#activeTaskCard {{ background: rgba(15, 28, 44, 0.94); border: {card_border}; border-radius: 18px; }}"
            )
            self._coding_workspace_card.setStyleSheet(
                f"QFrame#codingWorkspaceCard {{ background: rgba(35, 24, 11, 0.95); border: {card_border}; border-radius: 18px; }}"
            )
            self._final_answer_card.setStyleSheet(
                f"QFrame#finalAnswerCard {{ background: rgba(11, 24, 37, 0.96); border: {card_border}; border-radius: 18px; }}"
            )
            self._approval_title.setStyleSheet(f"font-size: 18px; font-weight: 700; color: {theme['highlight']};")
            self._approval_risk.setStyleSheet(f"font-size: 12px; font-weight: 700; color: {theme['warning']};")
            for widget in (self._approval_summary, self._approval_target, self._approval_reason):
                widget.setStyleSheet(f"font-size: 13px; color: {theme['text']};")
            self._long_horizon_title.setStyleSheet(f"font-size: 18px; font-weight: 700; color: {theme['highlight']};")
            self._long_horizon_phase.setStyleSheet(f"font-size: 12px; font-weight: 600; color: {theme['accent']};")
            self._long_horizon_summary.setStyleSheet(f"font-size: 13px; color: {theme['text']};")
            self._long_horizon_metrics.setStyleSheet(f"font-size: 12px; color: {theme['muted']};")
            self._long_horizon_delta.setStyleSheet(f"font-size: 12px; color: {theme['highlight']};")
            self._active_task_title.setStyleSheet(f"font-size: 18px; font-weight: 700; color: {theme['highlight']};")
            self._active_task_phase.setStyleSheet(f"font-size: 12px; font-weight: 600; color: {theme['accent']};")
            self._active_task_summary.setStyleSheet(f"font-size: 14px; color: {theme['text']};")
            self._active_task_metrics.setStyleSheet(f"font-size: 12px; color: {theme['muted']};")
            self._active_task_routes.setStyleSheet(f"font-size: 12px; color: {theme['highlight']};")
            self._active_task_warnings.setStyleSheet(f"font-size: 12px; color: {theme['warning']};")
            self._coding_workspace_title.setStyleSheet(f"font-size: 18px; font-weight: 700; color: {theme['highlight']};")
            self._coding_workspace_status.setStyleSheet(f"font-size: 12px; font-weight: 600; color: {theme['accent']};")
            self._coding_workspace_summary.setStyleSheet(f"font-size: 14px; color: {theme['text']};")
            self._coding_workspace_context.setStyleSheet(f"font-size: 12px; color: {theme['muted']};")
            self._coding_workspace_validation.setStyleSheet(f"font-size: 12px; color: {theme['highlight']};")
            self._coding_workspace_artifacts.setStyleSheet(f"font-size: 12px; color: {theme['text']};")
            self._coding_workspace_blockers.setStyleSheet(f"font-size: 12px; color: {theme['warning']};")
            self._final_answer_title.setStyleSheet(f"font-size: 18px; font-weight: 700; color: {theme['highlight']};")
            self._final_answer_status.setStyleSheet(f"font-size: 12px; font-weight: 600; color: {theme['success']};")
            self._final_answer_body.setStyleSheet(f"font-size: 14px; color: {theme['text']};")
            self._final_answer_evidence.setStyleSheet(f"font-size: 12px; color: {theme['muted']};")
            self._final_answer_refs.setStyleSheet(f"font-size: 12px; color: {theme['highlight']};")
            self._final_answer_warnings.setStyleSheet(f"font-size: 12px; color: {theme['warning']};")
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

        @QtCore.Slot()
        def shutdown(self) -> None:
            if self._timer.isActive():
                self._timer.stop()
            self._orb.shutdown()
            self.close()

        def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # pragma: no cover - exercised in UI tests
            if self._timer.isActive():
                self._timer.stop()
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
