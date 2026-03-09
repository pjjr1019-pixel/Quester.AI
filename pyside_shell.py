"""Premium PySide6 shell host for the local desktop UI migration."""

from __future__ import annotations

import math
import os
import threading
from collections.abc import Callable
from typing import Any

from data_structures import DashboardAppState, ShellState

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
            "amber_gold": ("#ffb85a", "#1a120b", "#fff4cb"),
            "focused_yellow": ("#ffd760", "#171208", "#fff8d8"),
            "deep_red": ("#ff646e", "#19080d", "#ffe0d5"),
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
            self._minimal = bool(ui.get("lightweight_mode", False)) or str(
                ui.get("shell_preset", "balanced")
            ).lower() == "minimal"
            self._reduced_motion = bool(ui.get("reduced_motion", False))
            self._timer.start(66 if self._reduced_motion else 50 if self._minimal else 33)

        def _on_tick(self) -> None:
            self._phase = (self._phase + (0.025 if self._reduced_motion else 0.04)) % (math.tau * 8.0)
            self.update()

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
            intensity = max(0.1, min(1.0, self._shell_state.orb_intensity))
            breathe = 1.0 + (math.sin(self._phase) * 0.04 * intensity)
            center = QtCore.QPointF(rect.center().x(), rect.center().y() - (rect.height() * 0.08))
            radius = min(rect.width(), rect.height()) * 0.21 * breathe

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

            if not self._minimal:
                tool_segments = max(1, min(8, len(self._shell_state.active_tools)))
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

            waveform_width = radius * 2.9
            waveform_y = center.y() + (radius * 1.62)
            waveform_pen = QtGui.QPen(QtGui.QColor(primary.red(), primary.green(), primary.blue(), int(120 * intensity)))
            waveform_pen.setWidthF(1.6)
            painter.setPen(waveform_pen)
            path = QtGui.QPainterPath()
            path.moveTo(center.x() - waveform_width / 2.0, waveform_y)
            points = 42
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
            request_action: Callable[[str, dict[str, Any] | None], bool] | None = None,
        ) -> None:
            super().__init__()
            self._shell_state = shell_state or ShellState()
            self._app_state = app_state or DashboardAppState()
            self._shell_state_provider = shell_state_provider
            self._app_state_provider = app_state_provider
            self._submit_task = submit_task
            self._request_action = request_action
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
            root.addWidget(self._activity_bar)
            root.addWidget(self._notification)
            self._conversation_scroll = QtWidgets.QScrollArea(central)
            self._conversation_scroll.setWidgetResizable(True)
            self._conversation_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
            self._conversation_container = QtWidgets.QWidget(self._conversation_scroll)
            self._conversation_layout = QtWidgets.QVBoxLayout(self._conversation_container)
            self._conversation_layout.setContentsMargins(0, 0, 0, 0)
            self._conversation_layout.setSpacing(12)
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
            self._fast = QtWidgets.QPushButton("Fast", dock)
            self._deep = QtWidgets.QPushButton("Deep", dock)
            self._long = QtWidgets.QPushButton("Long Horizon", dock)
            self._pause = QtWidgets.QPushButton("Pause", dock)
            self._resume = QtWidgets.QPushButton("Resume", dock)
            self._stop = QtWidgets.QPushButton("Stop", dock)
            self._fast.clicked.connect(lambda: self._slider.setValue(5))
            self._deep.clicked.connect(lambda: self._slider.setValue(30))
            self._long.clicked.connect(lambda: self._slider.setValue(180))
            self._pause.clicked.connect(lambda: self._dispatch_session_action("pause"))
            self._resume.clicked.connect(lambda: self._dispatch_session_action("resume"))
            self._stop.clicked.connect(lambda: self._dispatch_session_action("stop"))
            for widget in (self._thinking, self._slider, self._fast, self._deep, self._long, self._pause, self._resume, self._stop):
                bottom.addWidget(widget, 1 if widget is self._slider else 0)
            dock_layout.addLayout(top)
            dock_layout.addLayout(bottom)
            root.addWidget(dock)
            self.setCentralWidget(central)

            self._agent_text = QtWidgets.QPlainTextEdit(readOnly=True)
            self._timeline = QtWidgets.QListWidget()
            self._session_text = QtWidgets.QPlainTextEdit(readOnly=True)
            left_tabs = QtWidgets.QTabWidget(self)
            left_tabs.addTab(self._agent_text, "Agents")
            left_tabs.addTab(self._timeline, "Timeline")
            left_tabs.addTab(self._session_text, "Session")
            self._left_dock = QtWidgets.QDockWidget("Task Timeline", self)
            self._left_dock.setWidget(left_tabs)
            self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, self._left_dock)

            self._evidence = QtWidgets.QPlainTextEdit(readOnly=True)
            self._provenance = QtWidgets.QPlainTextEdit(readOnly=True)
            self._control_plane = QtWidgets.QPlainTextEdit(readOnly=True)
            self._runtime = QtWidgets.QPlainTextEdit(readOnly=True)
            right_tabs = QtWidgets.QTabWidget(self)
            right_tabs.addTab(self._evidence, "Evidence")
            right_tabs.addTab(self._provenance, "Provenance")
            right_tabs.addTab(self._control_plane, "Control Plane")
            right_tabs.addTab(self._runtime, "Runtime")
            self._right_dock = QtWidgets.QDockWidget("Evidence And Insights", self)
            self._right_dock.setWidget(right_tabs)
            self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self._right_dock)

            self._history = QtWidgets.QPlainTextEdit(readOnly=True)
            self._knowledge = QtWidgets.QPlainTextEdit(readOnly=True)
            self._settings = QtWidgets.QPlainTextEdit(readOnly=True)
            self._readiness = QtWidgets.QPlainTextEdit(readOnly=True)
            self._capability = QtWidgets.QPlainTextEdit(readOnly=True)
            self._debug = QtWidgets.QPlainTextEdit(readOnly=True)
            bottom_tabs = QtWidgets.QTabWidget(self)
            for name, widget in (("History", self._history), ("Knowledge", self._knowledge), ("Settings", self._settings), ("Readiness", self._readiness), ("Capabilities", self._capability), ("Debug", self._debug)):
                bottom_tabs.addTab(widget, name)
            self._bottom_dock = QtWidgets.QDockWidget("Secondary Views", self)
            self._bottom_dock.setWidget(bottom_tabs)
            self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, self._bottom_dock)

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
            self._notification.setText(shell_state.shell_notifications[-1].message if shell_state.shell_notifications else "")
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
            _clear(self._conversation_layout)
            items = shell_state.conversation_items or ()
            if not items:
                self._conversation_layout.addWidget(self._message_card("system", "Ready", shell_state.current_task_summary or "The shell is ready for a new task.", ()))
            else:
                for item in items:
                    self._conversation_layout.addWidget(self._message_card(item.role, item.title, item.body, item.chips))
            self._conversation_layout.addStretch(1)
            self._timeline.clear()
            for entry in shell_state.timeline_entries[-24:]:
                item = QtWidgets.QListWidgetItem(f"{entry.label}\n{entry.detail}".strip())
                item.setForeground(QtGui.QColor(theme["warning"] if entry.severity == "warning" else theme["danger"] if entry.severity == "error" else theme["text"]))
                self._timeline.addItem(item)
            self._apply_theme()

        def apply_dashboard_state(self, app_state: DashboardAppState) -> None:
            self._app_state = app_state
            self._slider.blockSignals(True)
            self._slider.setValue(max(1, min(720, int(app_state.user_settings.reasoning.get("thinking_minutes", 30) or 30))))
            self._slider.blockSignals(False)
            self._thinking.setText(f"Thinking: {self._slider.value()} min")
            self._orb.set_ui_preferences(app_state.user_settings.ui)
            self._mic.setEnabled("speech_to_text" in app_state.user_settings.models.get("enabled_roles", ()))
            self._agent_text.setPlainText("\n".join(f"{component}: {status.state.value} [{status.severity.value}] {status.message}" for component, status in sorted(app_state.statuses.items())) or "No live agent status yet.")
            session = app_state.local_task_session
            task = app_state.active_task
            self._session_text.setPlainText("\n".join((f"Session: {session.session_id or '(none)'}", f"Status: {session.status}", f"Target: {session.current_target or '(none)'}", f"Observation: {session.effective_observation_tier}", f"Approvals: {len(session.pending_approval_summaries)}", f"Long horizon: {task.long_horizon_session_id or '(none)'}", f"Cycles: {task.long_horizon_completed_cycles}/{task.long_horizon_total_cycles}")))
            self._evidence.setPlainText("\n".join((f"Local results: {task.local_result_count}", f"Web results: {task.web_result_count}", f"Used web fallback: {task.used_web_fallback}", f"Supporting IDs: {', '.join(task.supporting_evidence_ids) if task.supporting_evidence_ids else '(none)'}", f"Citations: {', '.join(task.citation_refs) if task.citation_refs else '(none)'}")))
            self._provenance.setPlainText("\n".join((f"Task id: {task.task_id or '(none)'}", f"Stage: {task.running_stage or app_state.last_stage or '(idle)'}", f"Verifier: {task.selected_verifier or '(none)'}", f"Candidate score: {task.candidate_score:.2f}", f"Repairs: {', '.join(task.repair_actions) if task.repair_actions else '(none)'}", f"Failures: {', '.join(task.failure_categories) if task.failure_categories else '(none)'}")))
            registry = app_state.model_registry_view
            route_lines = [
                f"- {decision.role.value}: {decision.selected_registration_key or '(none)'} | {decision.reason or '(no reason)'}"
                for decision in registry.last_route_decisions[:6]
            ]
            fallback_lines = [
                f"- {role}: {reason}"
                for role, reason in list(registry.fallback_reasons.items())[:6]
            ]
            optimizer_lines = [
                f"- {suggestion.kind.value}: {suggestion.summary or suggestion.suggestion_id}"
                for suggestion in registry.recent_optimizer_suggestions[:6]
            ]
            self._control_plane.setPlainText(
                "\n".join(
                    (
                        f"Installed roles: {len(registry.registrations)}",
                        f"Active heavy roles: {', '.join(registry.active_heavy_roles) if registry.active_heavy_roles else '(none)'}",
                        "",
                        "Recent route decisions:",
                        *(route_lines or ["- (none)"]),
                        "",
                        "Fallback reasons:",
                        *(fallback_lines or ["- (none)"]),
                        "",
                        "Optimizer suggestions:",
                        *(optimizer_lines or ["- (none)"]),
                    )
                )
            )
            health = app_state.runtime_health
            self._runtime.setPlainText("\n".join((f"Generation backend: {health.generation_backend or '(unknown)'}", f"Embedding backend: {health.embedding_backend or '(unknown)'}", f"Heavy slots: {len(health.active_heavy_roles)}/{health.heavy_slot_limit}", f"Governor active: {health.governor_active}", f"Governor summary: {health.governor_summary or '(none)'}", f"Fallback active: {health.fallback_active}", f"Fallback reason: {health.fallback_reason or '(none)'}", f"Last error: {health.last_error or '(none)'}")))
            self._history.setPlainText("\n".join(f"{entry.task_id} | {entry.question} | {entry.critique_result or 'pending'}" for entry in app_state.task_history[:20]) or "No task history loaded.")
            self._knowledge.setPlainText("\n".join(f"{source.source_ref} | {source.title} | {source.corpus_origin} | archived={source.archived}" for source in app_state.knowledge_sources[:40]) or "No knowledge sources loaded.")
            self._settings.setPlainText("\n".join((f"Profile: {app_state.user_settings.profile_name}", f"Shell: {app_state.user_settings.ui.get('app_shell', 'tkinter')}", f"Preset: {app_state.user_settings.ui.get('shell_preset', 'balanced')}", f"Reduced motion: {app_state.user_settings.ui.get('reduced_motion', False)}", f"Lightweight mode: {app_state.user_settings.ui.get('lightweight_mode', False)}", f"Reasoning mode: {app_state.user_settings.reasoning.get('mode', 'auto')}")))
            self._readiness.setPlainText("\n".join((f"Stub ready: {app_state.readiness_report.stub_mode_ready}", f"Real mode ready: {app_state.readiness_report.real_mode_ready}", "", *[f"{check.label}: {check.status} | {check.detail}" for check in app_state.readiness_report.checks])) or "Readiness has not been loaded yet.")
            self._capability.setPlainText("\n".join(f"{item.capability_name}: {item.status} | {item.reason}" for item in app_state.readiness_report.capabilities) or "Capability readiness has not been loaded yet.")
            self._debug.setPlainText("\n".join((f"last_stage: {app_state.last_stage}", f"event_count: {app_state.event_count}", f"dropped_events: {app_state.dropped_events}", f"last_notice: {app_state.last_notice}", f"summary: {self._shell_state.current_task_summary}")))
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
            self._ribbon.setText("   |   ".join(ribbon_parts))

        def _apply_theme(self) -> None:
            theme = _theme(self._shell_state)
            self.setStyleSheet(
                "QMainWindow, QWidget {"
                f"background: {theme['bg']}; color: {theme['text']};"
                "}"
                "QPlainTextEdit, QListWidget, QTabWidget::pane, QFrame {"
                f"background: {theme['panel']}; border: 1px solid {theme['edge']}; border-radius: 16px;"
                "}"
                "QLineEdit {"
                "background: rgba(10, 17, 29, 0.95);"
                f"color: {theme['text']}; border-radius: 18px; border: 1px solid {theme['edge']}; padding: 14px 18px;"
                "}"
                "QPushButton {"
                "background: rgba(255, 255, 255, 0.04);"
                f"color: {theme['text']}; border-radius: 14px; border: 1px solid {theme['edge']}; padding: 10px 14px;"
                "}"
                "QSlider::groove:horizontal { height: 6px; background: rgba(255,255,255,0.10); border-radius: 3px; }"
                f"QSlider::handle:horizontal {{ background: {theme['accent']}; width: 16px; margin: -6px 0; border-radius: 8px; }}"
                "QTabBar::tab { background: rgba(255,255,255,0.03); color: #b6c2d7; padding: 8px 12px; border-top-left-radius: 10px; border-top-right-radius: 10px; margin-right: 4px; }"
                "QTabBar::tab:selected { background: rgba(255,255,255,0.08); color: #eef5ff; }"
            )
            self._status.setStyleSheet(f"font-size: 34px; font-weight: 700; color: {theme['highlight']};")
            self._sub_status.setStyleSheet(f"font-size: 15px; color: {theme['muted']};")
            self._notification.setStyleSheet(f"font-size: 13px; color: {theme['warning'] if self._shell_state.approval_pending or self._shell_state.degraded_reason else theme['muted']};")
            self._ribbon.setStyleSheet(f"padding: 8px 12px; border-radius: 14px; background: rgba(255,255,255,0.03); border: 1px solid {theme['edge']}; color: {theme['muted']};")

        @QtCore.Slot()
        def shutdown(self) -> None:
            self.close()


    class PySideShellHost:
        """Threaded host that runs the PySide shell next to the asyncio backend."""

        def __init__(
            self,
            *,
            shell_state_provider: Callable[[], ShellState],
            app_state_provider: Callable[[], DashboardAppState],
            submit_task: Callable[[str, int], bool] | None = None,
            request_action: Callable[[str, dict[str, Any] | None], bool] | None = None,
            startup_timeout_s: float = 10.0,
        ) -> None:
            self._shell_state_provider = shell_state_provider
            self._app_state_provider = app_state_provider
            self._submit_task = submit_task
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
                    self._window.close()
                if self._app is not None:
                    self._app.processEvents()
                self._window = None
                self._app = None
                self._attached_mode = False
                return
            if self._window is not None:
                QtCore.QMetaObject.invokeMethod(self._window, "shutdown", QtCore.Qt.ConnectionType.QueuedConnection)
            if self._app is not None:
                QtCore.QMetaObject.invokeMethod(self._app, "quit", QtCore.Qt.ConnectionType.QueuedConnection)
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=timeout_s)

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
