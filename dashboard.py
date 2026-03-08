"""Local dashboard service with optional Tkinter UI."""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import replace
from typing import Any, Callable

from config import APP_CONFIG, AppConfig
from data_structures import (
    AgentStatus,
    AudioSynthesisResult,
    AudioTranscriptionResult,
    CodeSpecialistResult,
    DashboardAppState,
    DashboardCapabilityAvailability,
    DashboardKnowledgeSource,
    ModelRole,
    ModelRoleActionReport,
    ModelRegistryView,
    DashboardReadinessReport,
    DashboardTaskHistoryEntry,
    DashboardTaskInspector,
    DashboardRuntimeHealth,
    DemoPackStatus,
    RuntimeCondition,
    SampleTaskDefinition,
    TextTranslationResult,
    UserSettingsProfile,
    coerce_agent_status,
    coerce_audio_synthesis_result,
    coerce_audio_transcription_result,
    coerce_code_specialist_result,
    coerce_dashboard_capability_availability,
    coerce_dashboard_knowledge_source,
    coerce_dashboard_readiness_report,
    coerce_dashboard_task_history_entry,
    coerce_dashboard_task_inspector,
    coerce_model_registry_view,
    coerce_model_role_action_report,
    coerce_demo_pack_status,
    coerce_sample_task_definition,
    coerce_text_translation_result,
    coerce_user_settings_profile,
    utc_now,
)
from utils import utc_now_iso

try:
    import tkinter as tk
    from tkinter import ttk
except Exception:  # pragma: no cover - environment-specific
    tk = None
    ttk = None


class DashboardService:
    """Consumes typed events and optionally renders them in a local Tkinter window."""

    TIME_CONTROL_PRESETS: tuple[tuple[str, int], ...] = (
        ("Quick 1m", 1),
        ("Fast 5m", 5),
        ("Focus 30m", 30),
        ("Deep 120m", 120),
        ("Long 121m", 121),
        ("12h 720m", 720),
    )

    def __init__(self, config: AppConfig = APP_CONFIG):
        self.config = config
        self.logger = logging.getLogger("quester.dashboard")
        self._events: queue.Queue[dict[str, Any]] = queue.Queue(
            maxsize=config.concurrency.dashboard_queue_maxsize
        )
        self._max_debug_events_per_poll = max(
            1,
            min(config.concurrency.dashboard_queue_maxsize, 64),
        )
        self._started = False
        self._headless = not config.dashboard.enable_ui
        self._ui_thread: threading.Thread | None = None
        self._root: tk.Tk | None = None
        self._stop_flag = threading.Event()
        self._dropped_events = 0
        self._app_state = DashboardAppState()
        self._submit_task_callback: Callable[[str, int], None] | None = None
        self._save_settings_callback: Callable[[UserSettingsProfile], None] | None = None
        self._perform_action_callback: Callable[[str, dict[str, Any]], None] | None = None

        self._question_var: tk.StringVar | None = None
        self._thinking_minutes_var: tk.IntVar | None = None
        self._thinking_label_var: tk.StringVar | None = None
        self._time_summary_var: tk.StringVar | None = None
        self._run_status_var: tk.StringVar | None = None
        self._app_notice_var: tk.StringVar | None = None
        self._settings_status_var: tk.StringVar | None = None
        self._profile_name_var: tk.StringVar | None = None
        self._generation_backend_var: tk.StringVar | None = None
        self._embedding_backend_var: tk.StringVar | None = None
        self._vector_store_var: tk.StringVar | None = None
        self._web_provider_var: tk.StringVar | None = None
        self._reasoning_mode_var: tk.StringVar | None = None
        self._allow_web_fallback_var: tk.BooleanVar | None = None
        self._enable_self_optimizer_var: tk.BooleanVar | None = None
        self._reranking_var: tk.BooleanVar | None = None
        self._reranker_role_enabled_var: tk.BooleanVar | None = None
        self._speech_to_text_role_enabled_var: tk.BooleanVar | None = None
        self._text_to_speech_role_enabled_var: tk.BooleanVar | None = None
        self._vad_role_enabled_var: tk.BooleanVar | None = None
        self._translation_role_enabled_var: tk.BooleanVar | None = None
        self._code_specialist_role_enabled_var: tk.BooleanVar | None = None
        self._long_horizon_enabled_var: tk.BooleanVar | None = None
        self._long_horizon_minutes_var: tk.StringVar | None = None
        self._optimizer_policy_var: tk.StringVar | None = None
        self._optimizer_replay_limit_var: tk.StringVar | None = None
        self._show_debug_pane_var: tk.BooleanVar | None = None
        self._desktop_enabled_var: tk.BooleanVar | None = None
        self._desktop_approval_policy_var: tk.StringVar | None = None
        self._observation_tier_var: tk.StringVar | None = None
        self._cloud_mode_var: tk.StringVar | None = None
        self._log_runtime_events_var: tk.BooleanVar | None = None
        self._allow_cloud_content_var: tk.BooleanVar | None = None
        self._log_level_var: tk.StringVar | None = None
        self._settings_path_var: tk.StringVar | None = None
        self._history_task_id_var: tk.StringVar | None = None
        self._history_export_path_var: tk.StringVar | None = None
        self._knowledge_source_ref_var: tk.StringVar | None = None
        self._knowledge_title_var: tk.StringVar | None = None
        self._knowledge_status_var: tk.StringVar | None = None
        self._sample_task_id_var: tk.StringVar | None = None
        self._sample_export_path_var: tk.StringVar | None = None
        self._examples_status_var: tk.StringVar | None = None
        self._audio_path_var: tk.StringVar | None = None
        self._tts_text_var: tk.StringVar | None = None
        self._tts_output_path_var: tk.StringVar | None = None
        self._audio_status_var: tk.StringVar | None = None
        self._translation_source_language_var: tk.StringVar | None = None
        self._translation_target_language_var: tk.StringVar | None = None
        self._translation_status_var: tk.StringVar | None = None
        self._translation_input_var: tk.StringVar | None = None
        self._code_source_path_var: tk.StringVar | None = None
        self._code_request_var: tk.StringVar | None = None
        self._code_status_var: tk.StringVar | None = None
        self._model_role_var: tk.StringVar | None = None
        self._model_role_status_var: tk.StringVar | None = None

        self._answer_text: tk.Text | None = None
        self._citations_text: tk.Text | None = None
        self._critique_text: tk.Text | None = None
        self._evidence_text: tk.Text | None = None
        self._provenance_text: tk.Text | None = None
        self._web_text: tk.Text | None = None
        self._status_text: tk.Text | None = None
        self._health_text: tk.Text | None = None
        self._model_registry_text: tk.Text | None = None
        self._model_role_detail_text: tk.Text | None = None
        self._conditions_text: tk.Text | None = None
        self._long_horizon_text: tk.Text | None = None
        self._settings_profiles_text: tk.Text | None = None
        self._history_list_text: tk.Text | None = None
        self._history_detail_text: tk.Text | None = None
        self._knowledge_list_text: tk.Text | None = None
        self._knowledge_content_text: tk.Text | None = None
        self._examples_list_text: tk.Text | None = None
        self._examples_detail_text: tk.Text | None = None
        self._audio_text: tk.Text | None = None
        self._translation_input_text: tk.Text | None = None
        self._translation_output_text: tk.Text | None = None
        self._code_input_text: tk.Text | None = None
        self._code_output_text: tk.Text | None = None
        self._readiness_checks_text: tk.Text | None = None
        self._capability_text: tk.Text | None = None
        self._debug_text: tk.Text | None = None
        self._debug_frame: tk.Misc | None = None

    @property
    def dropped_events(self) -> int:
        """Return the number of queue-overflow drops observed since startup."""
        return self._dropped_events

    @property
    def ui_running(self) -> bool:
        """Report whether the Tkinter UI thread is currently live."""
        return bool(
            self._started
            and not self._headless
            and self._ui_thread is not None
            and self._ui_thread.is_alive()
            and not self._stop_flag.is_set()
        )

    def app_state_snapshot(self) -> DashboardAppState:
        """Return the latest typed app-state projection."""
        return self._app_state

    def attach_controller(
        self,
        *,
        submit_task: Callable[[str, int], None] | None = None,
        save_settings: Callable[[UserSettingsProfile], None] | None = None,
        perform_action: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        """Attach optional controller callbacks used by the Tkinter shell."""
        self._submit_task_callback = submit_task
        self._save_settings_callback = save_settings
        self._perform_action_callback = perform_action

    def apply_user_settings(self, profile: UserSettingsProfile | dict[str, Any]) -> None:
        """Update the typed UI state with a persisted settings profile."""
        settings_profile = coerce_user_settings_profile(profile)
        self._app_state = replace(
            self._app_state,
            user_settings=settings_profile,
            updated_at=utc_now(),
        )
        self._apply_settings_to_form(settings_profile)

    def request_task_submission(self, question: str, thinking_minutes: int) -> bool:
        """Submit a dashboard-triggered task request through the controller bridge."""
        normalized_question = question.strip()
        if not normalized_question or self._submit_task_callback is None:
            return False
        self._submit_task_callback(normalized_question, int(thinking_minutes))
        return True

    def request_settings_save(self, profile: UserSettingsProfile | dict[str, Any]) -> bool:
        """Persist a dashboard-triggered settings profile through the controller bridge."""
        settings_profile = coerce_user_settings_profile(profile)
        self.apply_user_settings(settings_profile)
        if self._save_settings_callback is None:
            return False
        self._save_settings_callback(settings_profile)
        return True

    def request_action(self, action: str, payload: dict[str, Any] | None = None) -> bool:
        """Invoke a non-task dashboard action through the controller bridge."""
        if self._perform_action_callback is None:
            return False
        self._perform_action_callback(action, dict(payload or {}))
        return True

    async def start(self) -> None:
        """Start the dashboard service."""
        if self._started:
            return
        self._started = True
        if self._headless or tk is None:
            self.logger.info("Dashboard running in headless mode.")
            return
        self._ui_thread = threading.Thread(target=self._run_ui, name="dashboard-ui", daemon=True)
        self._ui_thread.start()
        self.logger.info("Dashboard UI thread started.")

    async def stop(self) -> None:
        """Stop the dashboard service."""
        if not self._started:
            return
        self._started = False
        self._stop_flag.set()
        if self._root is not None:
            try:
                self._root.after(0, self._root.quit)
            except Exception:  # pragma: no cover - defensive
                pass
        if self._ui_thread and self._ui_thread.is_alive():
            self._ui_thread.join(timeout=self.config.preflight.flags.shutdown_timeout_s)
        self.logger.info("Dashboard stopped.")

    def publish_event(self, event: dict[str, Any]) -> None:
        """Publish event to queue for UI and diagnostics."""
        normalized = dict(event)
        if "timestamp" not in normalized:
            normalized["timestamp"] = utc_now_iso()
        final_event = normalized
        try:
            self._events.put_nowait(normalized)
        except queue.Full:
            try:
                # Preserve bounded memory by evicting the oldest debug event first.
                _ = self._events.get_nowait()
            except queue.Empty:
                pass
            self._dropped_events += 1
            final_event = dict(normalized)
            final_event["dropped_events"] = self._dropped_events
            final_event["queue_overflow"] = "evicted_oldest"
            try:
                self._events.put_nowait(final_event)
            except queue.Full:
                final_event["queue_overflow"] = "dropped_current_after_eviction"
        self._apply_event_to_state(final_event)

    def _run_ui(self) -> None:
        try:
            self._root = tk.Tk()
        except Exception as exc:  # pragma: no cover - depends on display availability
            self.logger.warning("Falling back to headless dashboard mode: %s", exc)
            self._headless = True
            return
        self._root.title(self.config.dashboard.window_title)
        self._root.geometry("1380x900")
        self._root.minsize(1100, 760)
        self._root.protocol("WM_DELETE_WINDOW", self._on_window_close)
        self._build_ui()
        self._apply_settings_to_form(self._app_state.user_settings)
        self._render_panels()
        self._schedule_poll()
        self._root.mainloop()
        self._stop_flag.set()

    def _build_ui(self) -> None:
        assert self._root is not None
        container = tk.Frame(self._root)
        container.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self._build_task_controls(container)

        body = tk.PanedWindow(container, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, sashwidth=6)
        body.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        left_column = tk.Frame(body)
        right_column = tk.Frame(body)
        body.add(left_column, stretch="always")
        body.add(right_column, stretch="always")

        self._answer_text = self._make_readonly_text(left_column, "Final Answer", height=8)
        self._citations_text = self._make_readonly_text(left_column, "Citations", height=4)
        self._critique_text = self._make_readonly_text(left_column, "Critique Summary", height=7)
        self._evidence_text = self._make_readonly_text(left_column, "Evidence Inspector", height=6)
        self._provenance_text = self._make_readonly_text(left_column, "Reasoning and Provenance", height=6)
        self._web_text = self._make_readonly_text(left_column, "Web Query Log", height=6)

        self._status_text = self._make_readonly_text(right_column, "Agent Status", height=8)
        self._health_text = self._make_readonly_text(right_column, "Runtime Health", height=8)
        self._model_registry_text = self._make_readonly_text(
            right_column,
            "Local AI Control Plane",
            height=8,
        )
        self._build_model_role_controls(right_column)
        self._model_role_detail_text = self._make_readonly_text(
            right_column,
            "Selected Local AI Role",
            height=8,
        )
        self._long_horizon_text = self._make_readonly_text(
            right_column,
            "Time Control and Long-Horizon Progress",
            height=10,
        )
        self._conditions_text = self._make_readonly_text(right_column, "Runtime Conditions", height=7)
        self._build_settings_panel(right_column)
        self._build_workspace_tabs(container)
        self._build_debug_panel(container)

    def _build_task_controls(self, parent: tk.Misc) -> None:
        frame = tk.LabelFrame(parent, text="Question and Run Control")
        frame.pack(fill=tk.X)

        self._question_var = tk.StringVar()
        self._thinking_minutes_var = tk.IntVar(value=30)
        self._thinking_label_var = tk.StringVar(value="30 minutes")
        self._time_summary_var = tk.StringVar(value="Planned mode: Interactive | 30 minute budget.")
        self._run_status_var = tk.StringVar(value="Ready.")
        self._app_notice_var = tk.StringVar(value="Local app shell ready.")

        tk.Label(frame, text="Question").grid(row=0, column=0, sticky="w")
        entry = tk.Entry(frame, textvariable=self._question_var, width=90)
        entry.grid(row=1, column=0, columnspan=4, sticky="ew", padx=(0, 8))

        tk.Label(frame, text="Thinking Time").grid(row=2, column=0, sticky="w", pady=(8, 0))
        scale = tk.Scale(
            frame,
            from_=1,
            to=self.config.budget_calibration.max_thinking_minutes,
            orient=tk.HORIZONTAL,
            showvalue=False,
            variable=self._thinking_minutes_var,
            command=self._on_thinking_minutes_changed,
            length=520,
        )
        scale.grid(row=3, column=0, columnspan=2, sticky="w")

        tk.Label(frame, textvariable=self._thinking_label_var).grid(row=3, column=2, sticky="w", padx=(8, 0))
        tk.Button(frame, text="Run Task", command=self._on_run_clicked, width=18).grid(
            row=3,
            column=3,
            sticky="e",
        )
        preset_row = tk.Frame(frame)
        preset_row.grid(row=4, column=0, columnspan=4, sticky="ew", pady=(6, 0))
        tk.Label(preset_row, text="Presets").pack(side="left")
        for label, minutes in self.TIME_CONTROL_PRESETS:
            tk.Button(
                preset_row,
                text=label,
                command=lambda value=minutes: self._on_apply_time_preset(value),
                width=10,
            ).pack(side="left", padx=(6, 0))

        action_row = tk.Frame(frame)
        action_row.grid(row=5, column=0, columnspan=4, sticky="ew", pady=(6, 0))
        tk.Button(action_row, text="Pause Session", command=self._on_pause_long_horizon_clicked, width=16).pack(
            side="left"
        )
        tk.Button(action_row, text="Resume Session", command=self._on_resume_long_horizon_clicked, width=16).pack(
            side="left",
            padx=(8, 0),
        )
        tk.Button(action_row, text="Cancel Session", command=self._on_cancel_long_horizon_clicked, width=16).pack(
            side="left",
            padx=(8, 0),
        )

        tk.Label(frame, textvariable=self._time_summary_var, anchor="w").grid(
            row=6,
            column=0,
            columnspan=4,
            sticky="ew",
            pady=(6, 0),
        )
        tk.Label(frame, textvariable=self._run_status_var, anchor="w").grid(
            row=7,
            column=0,
            columnspan=4,
            sticky="ew",
            pady=(4, 0),
        )
        tk.Label(frame, textvariable=self._app_notice_var, anchor="w").grid(
            row=8,
            column=0,
            columnspan=4,
            sticky="ew",
            pady=(4, 0),
        )

        frame.grid_columnconfigure(0, weight=1)
        entry.focus_set()

    def _build_model_role_controls(self, parent: tk.Misc) -> None:
        frame = tk.LabelFrame(parent, text="Local AI Role Actions")
        frame.pack(fill=tk.X, pady=(8, 0))

        self._model_role_var = tk.StringVar(value=ModelRole.GENERATION.value)
        self._model_role_status_var = tk.StringVar(value="Select a role to inspect or manage.")

        tk.Label(frame, text="Role").grid(row=0, column=0, sticky="w")
        tk.OptionMenu(frame, self._model_role_var, *[role.value for role in ModelRole]).grid(
            row=0,
            column=1,
            sticky="ew",
            padx=(8, 0),
        )

        action_row = tk.Frame(frame)
        action_row.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        tk.Button(action_row, text="Install Help", command=self._on_model_install_guidance_clicked, width=12).pack(
            side="left"
        )
        tk.Button(action_row, text="Enable", command=self._on_enable_model_role_clicked, width=10).pack(
            side="left",
            padx=(6, 0),
        )
        tk.Button(action_row, text="Disable", command=self._on_disable_model_role_clicked, width=10).pack(
            side="left",
            padx=(6, 0),
        )
        tk.Button(action_row, text="Warm", command=self._on_warm_model_role_clicked, width=10).pack(
            side="left",
            padx=(6, 0),
        )

        extra_row = tk.Frame(frame)
        extra_row.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        tk.Button(extra_row, text="Unload", command=self._on_unload_model_role_clicked, width=10).pack(
            side="left"
        )
        tk.Button(extra_row, text="Test Ping", command=self._on_test_model_role_clicked, width=10).pack(
            side="left",
            padx=(6, 0),
        )
        tk.Button(extra_row, text="Fallback", command=self._on_inspect_model_fallback_clicked, width=10).pack(
            side="left",
            padx=(6, 0),
        )

        tk.Label(frame, textvariable=self._model_role_status_var, anchor="w").grid(
            row=3,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(6, 0),
        )
        frame.grid_columnconfigure(1, weight=1)

    def _build_settings_panel(self, parent: tk.Misc) -> None:
        frame = tk.LabelFrame(parent, text="Settings")
        frame.pack(fill=tk.X, pady=(8, 0))

        self._settings_status_var = tk.StringVar(value="Settings loaded.")
        self._profile_name_var = tk.StringVar(value="default")
        self._generation_backend_var = tk.StringVar(value="ollama")
        self._embedding_backend_var = tk.StringVar(value="sentence_transformers")
        self._vector_store_var = tk.StringVar(value="chromadb")
        self._web_provider_var = tk.StringVar(value="wikipedia")
        self._reasoning_mode_var = tk.StringVar(value="auto")
        self._allow_web_fallback_var = tk.BooleanVar(value=True)
        self._enable_self_optimizer_var = tk.BooleanVar(value=False)
        self._reranking_var = tk.BooleanVar(value=True)
        self._reranker_role_enabled_var = tk.BooleanVar(value=False)
        self._speech_to_text_role_enabled_var = tk.BooleanVar(value=False)
        self._text_to_speech_role_enabled_var = tk.BooleanVar(value=False)
        self._vad_role_enabled_var = tk.BooleanVar(value=False)
        self._translation_role_enabled_var = tk.BooleanVar(value=False)
        self._code_specialist_role_enabled_var = tk.BooleanVar(value=False)
        self._long_horizon_enabled_var = tk.BooleanVar(value=False)
        self._long_horizon_minutes_var = tk.StringVar(value="120")
        self._optimizer_policy_var = tk.StringVar(value="proposal_only")
        self._optimizer_replay_limit_var = tk.StringVar(value="64")
        self._show_debug_pane_var = tk.BooleanVar(value=True)
        self._desktop_enabled_var = tk.BooleanVar(value=False)
        self._desktop_approval_policy_var = tk.StringVar(value="approve_risky_only")
        self._observation_tier_var = tk.StringVar(value="screenshot_on_demand")
        self._cloud_mode_var = tk.StringVar(value="auxiliary_only")
        self._log_runtime_events_var = tk.BooleanVar(value=True)
        self._allow_cloud_content_var = tk.BooleanVar(value=False)
        self._log_level_var = tk.StringVar(value="INFO")
        self._settings_path_var = tk.StringVar(value="logs/settings_profile.json")

        tk.Label(frame, text="Profile").grid(row=0, column=0, sticky="w")
        tk.Entry(frame, textvariable=self._profile_name_var, width=20).grid(
            row=0, column=1, sticky="ew", padx=(8, 0)
        )

        tk.Label(frame, text="Generation backend").grid(row=1, column=0, sticky="w", pady=(6, 0))
        tk.OptionMenu(frame, self._generation_backend_var, "ollama", "llama_cpp").grid(
            row=1,
            column=1,
            sticky="ew",
            padx=(8, 0),
            pady=(6, 0),
        )

        tk.Label(frame, text="Embedding backend").grid(row=2, column=0, sticky="w", pady=(6, 0))
        tk.OptionMenu(frame, self._embedding_backend_var, "sentence_transformers", "ollama_embeddings").grid(
            row=2,
            column=1,
            sticky="ew",
            padx=(8, 0),
            pady=(6, 0),
        )

        tk.Label(frame, text="Vector store").grid(row=3, column=0, sticky="w", pady=(6, 0))
        tk.OptionMenu(frame, self._vector_store_var, "chromadb", "simple_inmemory").grid(
            row=3,
            column=1,
            sticky="ew",
            padx=(8, 0),
            pady=(6, 0),
        )

        tk.Label(frame, text="Web provider").grid(row=4, column=0, sticky="w", pady=(6, 0))
        tk.OptionMenu(frame, self._web_provider_var, "wikipedia", "stub").grid(
            row=4,
            column=1,
            sticky="ew",
            padx=(8, 0),
            pady=(6, 0),
        )

        tk.Label(frame, text="Reasoning mode").grid(row=5, column=0, sticky="w", pady=(6, 0))
        tk.OptionMenu(frame, self._reasoning_mode_var, "auto", "fast", "deep").grid(
            row=5,
            column=1,
            sticky="ew",
            padx=(8, 0),
            pady=(6, 0),
        )

        tk.Label(frame, text="Observation tier").grid(row=6, column=0, sticky="w", pady=(6, 0))
        tk.OptionMenu(
            frame,
            self._observation_tier_var,
            "screenshot_on_demand",
            "ocr_on_step",
            "vision_on_step",
            "continuous_capture",
        ).grid(row=6, column=1, sticky="ew", padx=(8, 0), pady=(6, 0))

        tk.Label(frame, text="Cloud mode").grid(row=7, column=0, sticky="w", pady=(6, 0))
        tk.OptionMenu(frame, self._cloud_mode_var, "auxiliary_only", "disabled").grid(
            row=7,
            column=1,
            sticky="ew",
            padx=(8, 0),
            pady=(6, 0),
        )

        tk.Label(frame, text="Desktop approval").grid(row=8, column=0, sticky="w", pady=(6, 0))
        tk.OptionMenu(
            frame,
            self._desktop_approval_policy_var,
            "approve_risky_only",
            "manual_only",
            "safe_auto",
        ).grid(row=8, column=1, sticky="ew", padx=(8, 0), pady=(6, 0))

        tk.Label(frame, text="Optimizer policy").grid(row=9, column=0, sticky="w", pady=(6, 0))
        tk.OptionMenu(frame, self._optimizer_policy_var, "proposal_only").grid(
            row=9,
            column=1,
            sticky="ew",
            padx=(8, 0),
            pady=(6, 0),
        )

        tk.Label(frame, text="Long-horizon minutes").grid(row=10, column=0, sticky="w", pady=(6, 0))
        tk.Entry(frame, textvariable=self._long_horizon_minutes_var, width=12).grid(
            row=10, column=1, sticky="ew", padx=(8, 0), pady=(6, 0)
        )

        tk.Label(frame, text="Replay limit").grid(row=11, column=0, sticky="w", pady=(6, 0))
        tk.Entry(frame, textvariable=self._optimizer_replay_limit_var, width=12).grid(
            row=11, column=1, sticky="ew", padx=(8, 0), pady=(6, 0)
        )

        tk.Label(frame, text="Log level").grid(row=12, column=0, sticky="w", pady=(6, 0))
        tk.OptionMenu(frame, self._log_level_var, "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL").grid(
            row=12,
            column=1,
            sticky="ew",
            padx=(8, 0),
            pady=(6, 0),
        )

        tk.Checkbutton(frame, text="Allow web fallback", variable=self._allow_web_fallback_var).grid(
            row=13, column=0, columnspan=2, sticky="w", pady=(6, 0)
        )
        tk.Checkbutton(frame, text="Enable self-optimizer", variable=self._enable_self_optimizer_var).grid(
            row=14, column=0, columnspan=2, sticky="w"
        )
        tk.Checkbutton(frame, text="Enable reranking", variable=self._reranking_var).grid(
            row=15, column=0, columnspan=2, sticky="w"
        )
        tk.Checkbutton(frame, text="Enable reranker role", variable=self._reranker_role_enabled_var).grid(
            row=16, column=0, columnspan=2, sticky="w"
        )
        tk.Checkbutton(frame, text="Enable speech-to-text role", variable=self._speech_to_text_role_enabled_var).grid(
            row=17, column=0, columnspan=2, sticky="w"
        )
        tk.Checkbutton(frame, text="Enable text-to-speech role", variable=self._text_to_speech_role_enabled_var).grid(
            row=18, column=0, columnspan=2, sticky="w"
        )
        tk.Checkbutton(frame, text="Enable VAD role", variable=self._vad_role_enabled_var).grid(
            row=19, column=0, columnspan=2, sticky="w"
        )
        tk.Checkbutton(frame, text="Enable translation role", variable=self._translation_role_enabled_var).grid(
            row=20, column=0, columnspan=2, sticky="w"
        )
        tk.Checkbutton(frame, text="Enable code-specialist role", variable=self._code_specialist_role_enabled_var).grid(
            row=21, column=0, columnspan=2, sticky="w"
        )
        tk.Checkbutton(frame, text="Enable long-horizon mode", variable=self._long_horizon_enabled_var).grid(
            row=22, column=0, columnspan=2, sticky="w"
        )
        tk.Checkbutton(frame, text="Show debug pane", variable=self._show_debug_pane_var).grid(
            row=23, column=0, columnspan=2, sticky="w"
        )
        tk.Checkbutton(frame, text="Desktop mode enabled", variable=self._desktop_enabled_var).grid(
            row=24, column=0, columnspan=2, sticky="w"
        )
        tk.Checkbutton(frame, text="Log runtime events", variable=self._log_runtime_events_var).grid(
            row=25, column=0, columnspan=2, sticky="w"
        )
        tk.Checkbutton(frame, text="Allow cloud content", variable=self._allow_cloud_content_var).grid(
            row=26, column=0, columnspan=2, sticky="w"
        )

        tk.Label(frame, text="Profile import/export path").grid(row=27, column=0, sticky="w", pady=(6, 0))
        tk.Entry(frame, textvariable=self._settings_path_var, width=24).grid(
            row=27, column=1, sticky="ew", padx=(8, 0), pady=(6, 0)
        )

        button_row = tk.Frame(frame)
        button_row.grid(row=28, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        tk.Button(button_row, text="Save Profile", command=self._on_save_settings_clicked, width=16).pack(
            side="left"
        )
        tk.Button(button_row, text="Load Saved", command=self._on_load_settings_clicked, width=16).pack(
            side="left",
            padx=(8, 0),
        )
        tk.Button(button_row, text="Reset Form", command=self._on_reset_settings_clicked, width=16).pack(
            side="left",
            padx=(8, 0),
        )

        extra_row = tk.Frame(frame)
        extra_row.grid(row=29, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        tk.Button(extra_row, text="Refresh Profiles", command=self._on_refresh_settings_profiles_clicked, width=16).pack(
            side="left"
        )
        tk.Button(extra_row, text="Import JSON", command=self._on_import_settings_clicked, width=16).pack(
            side="left",
            padx=(8, 0),
        )
        tk.Button(extra_row, text="Export JSON", command=self._on_export_settings_clicked, width=16).pack(
            side="left",
            padx=(8, 0),
        )

        profiles_frame = tk.LabelFrame(frame, text="Saved Profiles")
        profiles_frame.grid(row=30, column=0, columnspan=2, sticky="nsew", pady=(8, 0))
        self._settings_profiles_text = tk.Text(profiles_frame, height=4, width=32, wrap="word")
        self._settings_profiles_text.pack(fill=tk.BOTH, expand=True)
        tk.Label(frame, textvariable=self._settings_status_var, anchor="w").grid(
            row=31,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(6, 0),
        )

        frame.grid_columnconfigure(1, weight=1)

    def _build_debug_panel(self, parent: tk.Misc) -> None:
        self._debug_frame = tk.LabelFrame(parent, text="Raw Event Log")
        self._debug_frame.pack(fill=tk.BOTH, expand=False, pady=(8, 0))
        self._debug_text = tk.Text(self._debug_frame, height=12, width=120, wrap="word")
        self._debug_text.pack(fill=tk.BOTH, expand=True)

    def _build_workspace_tabs(self, parent: tk.Misc) -> None:
        if ttk is None:
            self._build_examples_tab(parent)
            self._build_history_tab(parent)
            self._build_knowledge_tab(parent)
            self._build_audio_tab(parent)
            self._build_translation_tab(parent)
            self._build_code_tab(parent)
            self._build_readiness_tab(parent)
            return
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        examples_tab = tk.Frame(notebook)
        history_tab = tk.Frame(notebook)
        knowledge_tab = tk.Frame(notebook)
        audio_tab = tk.Frame(notebook)
        translation_tab = tk.Frame(notebook)
        code_tab = tk.Frame(notebook)
        readiness_tab = tk.Frame(notebook)
        notebook.add(examples_tab, text="Examples")
        notebook.add(history_tab, text="History")
        notebook.add(knowledge_tab, text="Knowledge")
        notebook.add(audio_tab, text="Audio")
        notebook.add(translation_tab, text="Translation")
        notebook.add(code_tab, text="Code")
        notebook.add(readiness_tab, text="Readiness")

        self._build_examples_tab(examples_tab)
        self._build_history_tab(history_tab)
        self._build_knowledge_tab(knowledge_tab)
        self._build_audio_tab(audio_tab)
        self._build_translation_tab(translation_tab)
        self._build_code_tab(code_tab)
        self._build_readiness_tab(readiness_tab)

    def _build_examples_tab(self, parent: tk.Misc) -> None:
        controls = tk.LabelFrame(parent, text="Phase 11 Examples")
        controls.pack(fill=tk.X, pady=(0, 8))

        self._sample_task_id_var = tk.StringVar(value="")
        self._sample_export_path_var = tk.StringVar(
            value="logs/dashboard_exports/phase11_verified_trace_export.jsonl"
        )
        self._examples_status_var = tk.StringVar(value="Phase 11 demo pack available.")
        tk.Label(controls, text="Sample ID").grid(row=0, column=0, sticky="w")
        tk.Entry(controls, textvariable=self._sample_task_id_var, width=32).grid(
            row=0, column=1, sticky="ew", padx=(8, 0)
        )
        tk.Label(controls, text="Export path").grid(row=1, column=0, sticky="w", pady=(6, 0))
        tk.Entry(controls, textvariable=self._sample_export_path_var, width=42).grid(
            row=1, column=1, sticky="ew", padx=(8, 0), pady=(6, 0)
        )
        button_row = tk.Frame(controls)
        button_row.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        tk.Button(button_row, text="Refresh Examples", command=self._on_refresh_examples_clicked, width=16).pack(
            side="left"
        )
        tk.Button(button_row, text="Load Demo Pack", command=self._on_load_demo_pack_clicked, width=16).pack(
            side="left",
            padx=(8, 0),
        )
        tk.Button(button_row, text="Load Sample", command=self._on_load_sample_clicked, width=14).pack(
            side="left",
            padx=(8, 0),
        )
        tk.Button(button_row, text="Run Sample", command=self._on_run_sample_clicked, width=14).pack(
            side="left",
            padx=(8, 0),
        )
        tk.Button(
            button_row,
            text="Export Verified Trace",
            command=self._on_export_verified_trace_clicked,
            width=20,
        ).pack(side="left", padx=(8, 0))
        tk.Label(controls, textvariable=self._examples_status_var, anchor="w").grid(
            row=3,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(6, 0),
        )
        controls.grid_columnconfigure(1, weight=1)

        self._examples_list_text = self._make_readonly_text(parent, "Sample Tasks", height=10)
        self._examples_detail_text = self._make_readonly_text(parent, "Sample Detail", height=12)

    def _build_history_tab(self, parent: tk.Misc) -> None:
        controls = tk.LabelFrame(parent, text="Task History Controls")
        controls.pack(fill=tk.X, pady=(0, 8))

        self._history_task_id_var = tk.StringVar(value="")
        self._history_export_path_var = tk.StringVar(value="logs/dashboard_exports/task_trace_debug.md")
        tk.Label(controls, text="Task ID").grid(row=0, column=0, sticky="w")
        tk.Entry(controls, textvariable=self._history_task_id_var, width=28).grid(
            row=0, column=1, sticky="ew", padx=(8, 0)
        )
        tk.Label(controls, text="Export path").grid(row=1, column=0, sticky="w", pady=(6, 0))
        tk.Entry(controls, textvariable=self._history_export_path_var, width=36).grid(
            row=1, column=1, sticky="ew", padx=(8, 0), pady=(6, 0)
        )
        button_row = tk.Frame(controls)
        button_row.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        tk.Button(button_row, text="Refresh History", command=self._on_refresh_history_clicked, width=16).pack(
            side="left"
        )
        tk.Button(button_row, text="Inspect Task", command=self._on_inspect_task_clicked, width=16).pack(
            side="left",
            padx=(8, 0),
        )
        tk.Button(
            button_row,
            text="Export Trace Debug",
            command=self._on_export_task_debug_clicked,
            width=18,
        ).pack(side="left", padx=(8, 0))
        controls.grid_columnconfigure(1, weight=1)

        self._history_list_text = self._make_readonly_text(parent, "Recent Tasks", height=9)
        self._history_detail_text = self._make_readonly_text(parent, "Run Inspector", height=12)

    def _build_knowledge_tab(self, parent: tk.Misc) -> None:
        controls = tk.LabelFrame(parent, text="Knowledge Library Controls")
        controls.pack(fill=tk.X, pady=(0, 8))

        self._knowledge_source_ref_var = tk.StringVar(value="")
        self._knowledge_title_var = tk.StringVar(value="")
        self._knowledge_status_var = tk.StringVar(value="Knowledge library ready.")
        tk.Label(controls, text="Source ref").grid(row=0, column=0, sticky="w")
        tk.Entry(controls, textvariable=self._knowledge_source_ref_var, width=32).grid(
            row=0, column=1, sticky="ew", padx=(8, 0)
        )
        tk.Label(controls, text="Title").grid(row=1, column=0, sticky="w", pady=(6, 0))
        tk.Entry(controls, textvariable=self._knowledge_title_var, width=32).grid(
            row=1, column=1, sticky="ew", padx=(8, 0), pady=(6, 0)
        )
        button_row = tk.Frame(controls)
        button_row.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        tk.Button(button_row, text="Refresh", command=self._on_refresh_knowledge_clicked, width=14).pack(side="left")
        tk.Button(button_row, text="Ingest Text", command=self._on_ingest_knowledge_clicked, width=14).pack(
            side="left",
            padx=(8, 0),
        )
        tk.Button(button_row, text="Archive", command=self._on_archive_knowledge_clicked, width=12).pack(
            side="left",
            padx=(8, 0),
        )
        tk.Button(button_row, text="Unarchive", command=self._on_unarchive_knowledge_clicked, width=12).pack(
            side="left",
            padx=(8, 0),
        )
        tk.Button(button_row, text="Rebuild", command=self._on_rebuild_knowledge_clicked, width=12).pack(
            side="left",
            padx=(8, 0),
        )
        tk.Button(button_row, text="Remove", command=self._on_remove_knowledge_clicked, width=12).pack(
            side="left",
            padx=(8, 0),
        )
        tk.Label(controls, textvariable=self._knowledge_status_var, anchor="w").grid(
            row=3,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(6, 0),
        )
        controls.grid_columnconfigure(1, weight=1)

        self._knowledge_list_text = self._make_readonly_text(parent, "Knowledge Sources", height=8)
        ingest_frame = tk.LabelFrame(parent, text="Document Ingest Text")
        ingest_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        self._knowledge_content_text = tk.Text(ingest_frame, height=10, width=80, wrap="word")
        self._knowledge_content_text.pack(fill=tk.BOTH, expand=True)

    def _build_readiness_tab(self, parent: tk.Misc) -> None:
        controls = tk.LabelFrame(parent, text="Readiness")
        controls.pack(fill=tk.X, pady=(0, 8))
        tk.Button(controls, text="Refresh Readiness", command=self._on_refresh_readiness_clicked, width=18).pack(
            side="left"
        )
        self._readiness_checks_text = self._make_readonly_text(parent, "Preflight Checks", height=10)
        self._capability_text = self._make_readonly_text(parent, "Capability Gates", height=10)

    def _build_audio_tab(self, parent: tk.Misc) -> None:
        input_controls = tk.LabelFrame(parent, text="Voice Input")
        input_controls.pack(fill=tk.X, pady=(0, 8))

        self._audio_path_var = tk.StringVar(value="logs/audio_input.wav")
        self._tts_text_var = tk.StringVar(value="")
        self._tts_output_path_var = tk.StringVar(value="logs/audio_output.wav")
        self._audio_status_var = tk.StringVar(value="Voice input and output ready.")
        tk.Label(input_controls, text="Audio file").grid(row=0, column=0, sticky="w")
        tk.Entry(input_controls, textvariable=self._audio_path_var, width=48).grid(
            row=0, column=1, sticky="ew", padx=(8, 0)
        )
        input_button_row = tk.Frame(input_controls)
        input_button_row.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        tk.Button(
            input_button_row,
            text="Transcribe Audio",
            command=self._on_transcribe_audio_clicked,
            width=16,
        ).pack(side="left")
        tk.Button(
            input_button_row,
            text="Use Transcript",
            command=self._on_use_audio_transcript_clicked,
            width=16,
        ).pack(side="left", padx=(8, 0))
        tk.Button(input_button_row, text="Clear", command=self._on_clear_audio_clicked, width=12).pack(
            side="left",
            padx=(8, 0),
        )
        input_controls.grid_columnconfigure(1, weight=1)

        output_controls = tk.LabelFrame(parent, text="Voice Output")
        output_controls.pack(fill=tk.X, pady=(0, 8))
        tk.Label(output_controls, text="Text to speak").grid(row=0, column=0, sticky="w")
        tk.Entry(output_controls, textvariable=self._tts_text_var, width=48).grid(
            row=0,
            column=1,
            sticky="ew",
            padx=(8, 0),
        )
        tk.Label(output_controls, text="Output file").grid(row=1, column=0, sticky="w", pady=(6, 0))
        tk.Entry(output_controls, textvariable=self._tts_output_path_var, width=48).grid(
            row=1,
            column=1,
            sticky="ew",
            padx=(8, 0),
            pady=(6, 0),
        )
        output_button_row = tk.Frame(output_controls)
        output_button_row.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        tk.Button(output_button_row, text="Speak Text", command=self._on_synthesize_audio_text_clicked, width=16).pack(
            side="left"
        )
        tk.Button(output_button_row, text="Speak Answer", command=self._on_speak_answer_clicked, width=16).pack(
            side="left",
            padx=(8, 0),
        )
        tk.Button(output_button_row, text="Clear Output", command=self._on_clear_audio_output_clicked, width=14).pack(
            side="left",
            padx=(8, 0),
        )
        tk.Label(output_controls, textvariable=self._audio_status_var, anchor="w").grid(
            row=3,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(6, 0),
        )
        output_controls.grid_columnconfigure(1, weight=1)
        self._audio_text = self._make_readonly_text(parent, "Audio Input and Output", height=14)

    def _build_translation_tab(self, parent: tk.Misc) -> None:
        controls = tk.LabelFrame(parent, text="Local Translation")
        controls.pack(fill=tk.X, pady=(0, 8))

        self._translation_source_language_var = tk.StringVar(value=self.config.translation.default_source_language)
        self._translation_target_language_var = tk.StringVar(value=self.config.translation.default_target_language)
        self._translation_input_var = tk.StringVar(value="")
        self._translation_status_var = tk.StringVar(value="Translation role ready.")
        tk.Label(controls, text="Source language").grid(row=0, column=0, sticky="w")
        tk.Entry(controls, textvariable=self._translation_source_language_var, width=12).grid(
            row=0, column=1, sticky="w", padx=(8, 0)
        )
        tk.Label(controls, text="Target language").grid(row=0, column=2, sticky="w", padx=(12, 0))
        tk.Entry(controls, textvariable=self._translation_target_language_var, width=12).grid(
            row=0, column=3, sticky="w", padx=(8, 0)
        )
        tk.Label(controls, text="Input text").grid(row=1, column=0, sticky="w", pady=(6, 0))
        tk.Entry(controls, textvariable=self._translation_input_var, width=72).grid(
            row=1, column=1, columnspan=3, sticky="ew", padx=(8, 0), pady=(6, 0)
        )
        button_row = tk.Frame(controls)
        button_row.grid(row=2, column=0, columnspan=4, sticky="ew", pady=(8, 0))
        tk.Button(button_row, text="Translate Text", command=self._on_translate_text_clicked, width=16).pack(
            side="left"
        )
        tk.Button(button_row, text="Translate Answer", command=self._on_translate_answer_clicked, width=16).pack(
            side="left",
            padx=(8, 0),
        )
        tk.Button(button_row, text="Use as Question", command=self._on_use_translation_clicked, width=16).pack(
            side="left",
            padx=(8, 0),
        )
        tk.Button(button_row, text="Clear", command=self._on_clear_translation_clicked, width=12).pack(
            side="left",
            padx=(8, 0),
        )
        tk.Label(controls, textvariable=self._translation_status_var, anchor="w").grid(
            row=3,
            column=0,
            columnspan=4,
            sticky="ew",
            pady=(6, 0),
        )
        controls.grid_columnconfigure(3, weight=1)
        self._translation_output_text = self._make_readonly_text(parent, "Translation Output", height=14)

    def _build_code_tab(self, parent: tk.Misc) -> None:
        controls = tk.LabelFrame(parent, text="Code Specialist")
        controls.pack(fill=tk.X, pady=(0, 8))

        self._code_source_path_var = tk.StringVar(value="orchestrator.py")
        self._code_request_var = tk.StringVar(value=self.config.code_specialist.default_request)
        self._code_status_var = tk.StringVar(value="Code specialist ready.")
        tk.Label(controls, text="Source path").grid(row=0, column=0, sticky="w")
        tk.Entry(controls, textvariable=self._code_source_path_var, width=56).grid(
            row=0, column=1, sticky="ew", padx=(8, 0)
        )
        tk.Label(controls, text="Request").grid(row=1, column=0, sticky="w", pady=(6, 0))
        tk.Entry(controls, textvariable=self._code_request_var, width=72).grid(
            row=1, column=1, sticky="ew", padx=(8, 0), pady=(6, 0)
        )
        button_row = tk.Frame(controls)
        button_row.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        tk.Button(button_row, text="Analyze File", command=self._on_analyze_code_file_clicked, width=16).pack(
            side="left"
        )
        tk.Button(button_row, text="Analyze Snippet", command=self._on_analyze_code_snippet_clicked, width=16).pack(
            side="left",
            padx=(8, 0),
        )
        tk.Button(button_row, text="Clear", command=self._on_clear_code_clicked, width=12).pack(
            side="left",
            padx=(8, 0),
        )
        tk.Label(controls, textvariable=self._code_status_var, anchor="w").grid(
            row=3,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(6, 0),
        )
        controls.grid_columnconfigure(1, weight=1)
        snippet_frame = tk.LabelFrame(parent, text="Code Snippet")
        snippet_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        self._code_input_text = tk.Text(snippet_frame, height=10, width=80, wrap="none")
        self._code_input_text.pack(fill=tk.BOTH, expand=True)
        self._code_output_text = self._make_readonly_text(parent, "Code Specialist Output", height=12)

    def _make_readonly_text(self, parent: tk.Misc, title: str, *, height: int) -> tk.Text:
        frame = tk.LabelFrame(parent, text=title)
        frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        widget = tk.Text(frame, height=height, width=80, wrap="word")
        widget.pack(fill=tk.BOTH, expand=True)
        return widget

    def _schedule_poll(self) -> None:
        if self._root is None:
            return
        self._drain_queue()
        self._render_panels()
        if not self._stop_flag.is_set():
            self._root.after(self.config.dashboard.refresh_interval_ms, self._schedule_poll)

    def _drain_queue(self) -> None:
        if self._debug_text is None:
            return
        for _ in range(self._max_debug_events_per_poll):
            try:
                event = self._events.get_nowait()
            except queue.Empty:
                break
            self._debug_text.insert("end", f"{event.get('timestamp')} | {event}\n")
            self._debug_text.see("end")

    def _apply_event_to_state(self, event: dict[str, Any]) -> None:
        state = self._app_state
        active_task = state.active_task
        runtime_health = state.runtime_health
        statuses = dict(state.statuses)
        recent_conditions = list(state.recent_conditions)
        settings_profiles = state.settings_profiles
        task_history = state.task_history
        selected_task = state.selected_task
        knowledge_sources = state.knowledge_sources
        readiness_report = state.readiness_report
        model_registry_view = state.model_registry_view
        model_role_action = state.model_role_action
        demo_pack_status = state.demo_pack_status
        sample_tasks = state.sample_tasks
        selected_sample_task = state.selected_sample_task
        audio_input = state.audio_input
        audio_output = state.audio_output
        translation_output = state.translation_output
        code_output = state.code_output
        last_notice = state.last_notice
        last_notice_severity = state.last_notice_severity

        stage = str(event.get("stage", ""))
        task_id = str(event.get("task_id", active_task.task_id))
        if task_id:
            active_task = replace(active_task, task_id=task_id)

        if stage == "status.updated":
            status = coerce_agent_status(event)
            statuses[status.component] = status
            if status.task_id:
                active_task = replace(active_task, task_id=status.task_id)
        elif stage == "pipeline.received":
            raw_budget = event.get("budget", {})
            budget = dict(raw_budget) if isinstance(raw_budget, dict) else {}
            requested_minutes = int(event.get("thinking_minutes", 0))
            planned_cycles = int(budget.get("planned_cycles", 1) or 1)
            long_horizon_mode = bool(active_task.long_horizon_session_id) or planned_cycles > 1
            active_task = replace(
                active_task,
                question=str(event.get("question", "")),
                thinking_minutes=(
                    active_task.requested_thinking_minutes or requested_minutes
                    if long_horizon_mode
                    else requested_minutes
                ),
                requested_thinking_minutes=(
                    active_task.requested_thinking_minutes or requested_minutes
                    if long_horizon_mode
                    else requested_minutes
                ),
                execution_mode="long_horizon" if long_horizon_mode else "interactive",
                long_horizon_cycle_budget_minutes=(
                    int(budget.get("cycle_budget_minutes", active_task.long_horizon_cycle_budget_minutes) or 0)
                    if long_horizon_mode
                    else active_task.long_horizon_cycle_budget_minutes
                ),
                running_stage=stage,
                updated_at=utc_now(),
            )
        elif stage == "pipeline.researcher_done":
            active_task = replace(
                active_task,
                running_stage=stage,
                local_result_count=int(event.get("local_result_count", 0)),
                web_result_count=int(event.get("web_result_count", 0)),
                used_web_fallback=bool(event.get("used_web_fallback", False)),
                web_source_refs=tuple(str(item) for item in event.get("web_source_refs", ())),
                updated_at=utc_now(),
            )
        elif stage == "researcher.web_lookup":
            active_task = replace(
                active_task,
                web_query=str(event.get("query", "")),
                web_source_refs=tuple(str(item) for item in event.get("source_refs", ())),
                used_web_fallback=True,
                web_result_count=int(event.get("persisted_results", active_task.web_result_count)),
                updated_at=utc_now(),
            )
        elif stage == "researcher.local_lookup":
            specialist_roles_used = list(active_task.specialist_roles_used)
            specialist_explanations = list(active_task.specialist_role_explanations)
            reranker_used = bool(event.get("specialist_reranker_used", False))
            reranker_backend = str(event.get("specialist_reranker_backend", "")).strip()
            reranker_reason = str(event.get("specialist_reranker_reason", "")).strip()
            if reranker_used and "reranker" not in specialist_roles_used:
                specialist_roles_used.append("reranker")
            specialist_explanations.append(
                (
                    f"reranker used via {reranker_backend or '(unknown)'} during local retrieval"
                    if reranker_used
                    else f"reranker not used: {reranker_reason or 'not_needed_or_disabled'}"
                )
            )
            active_task = replace(
                active_task,
                specialist_roles_used=tuple(dict.fromkeys(specialist_roles_used)),
                specialist_role_explanations=tuple(dict.fromkeys(specialist_explanations)),
                updated_at=utc_now(),
            )
        if stage.startswith("pipeline.long_horizon_"):
            active_task = self._merge_long_horizon_event_metrics(active_task, event)
        if stage == "pipeline.long_horizon_started":
            active_task = replace(
                active_task,
                long_horizon_session_id=str(event.get("session_id", "")),
                long_horizon_status="running",
                long_horizon_completed_cycles=0,
                long_horizon_total_cycles=int(event.get("planned_cycles", 0)),
                long_horizon_resume_count=0,
                long_horizon_pause_requested=False,
                long_horizon_cancel_requested=False,
                long_horizon_throttled=False,
                long_horizon_throttle_reason="",
                running_stage=stage,
                updated_at=utc_now(),
            )
        elif stage == "pipeline.long_horizon_pause_requested":
            active_task = replace(
                active_task,
                long_horizon_session_id=str(event.get("session_id", active_task.long_horizon_session_id)),
                long_horizon_current_phase="pause_requested",
                long_horizon_pause_requested=True,
                long_horizon_cancel_requested=False,
                updated_at=utc_now(),
            )
        elif stage == "pipeline.long_horizon_cancel_requested":
            active_task = replace(
                active_task,
                long_horizon_session_id=str(event.get("session_id", active_task.long_horizon_session_id)),
                long_horizon_current_phase="cancel_requested",
                long_horizon_pause_requested=False,
                long_horizon_cancel_requested=True,
                updated_at=utc_now(),
            )
        elif stage == "pipeline.long_horizon_resumed":
            active_task = replace(
                active_task,
                long_horizon_session_id=str(event.get("session_id", active_task.long_horizon_session_id)),
                long_horizon_status="running",
                long_horizon_completed_cycles=int(
                    event.get("completed_cycles", active_task.long_horizon_completed_cycles)
                ),
                long_horizon_total_cycles=int(event.get("total_cycles", active_task.long_horizon_total_cycles)),
                long_horizon_resume_count=int(event.get("resume_count", active_task.long_horizon_resume_count)),
                long_horizon_pause_requested=False,
                long_horizon_cancel_requested=False,
                running_stage=stage,
                updated_at=utc_now(),
            )
        elif stage == "pipeline.long_horizon_cycle_started":
            active_task = replace(
                active_task,
                long_horizon_session_id=str(event.get("session_id", active_task.long_horizon_session_id)),
                long_horizon_status="running",
                long_horizon_total_cycles=int(event.get("total_cycles", active_task.long_horizon_total_cycles)),
                long_horizon_resume_count=int(event.get("resume_count", active_task.long_horizon_resume_count)),
                long_horizon_throttled=bool(event.get("throttled", active_task.long_horizon_throttled)),
                long_horizon_throttle_reason=str(
                    event.get("throttle_reason", active_task.long_horizon_throttle_reason)
                ),
                running_stage=stage,
                updated_at=utc_now(),
            )
        elif stage == "pipeline.long_horizon_cycle_completed":
            active_task = replace(
                active_task,
                long_horizon_session_id=str(event.get("session_id", active_task.long_horizon_session_id)),
                long_horizon_status="checkpointed",
                long_horizon_completed_cycles=int(
                    event.get("cycle_index", active_task.long_horizon_completed_cycles)
                ),
                long_horizon_total_cycles=int(event.get("total_cycles", active_task.long_horizon_total_cycles)),
                long_horizon_resume_count=int(event.get("resume_count", active_task.long_horizon_resume_count)),
                long_horizon_throttled=bool(event.get("throttled", active_task.long_horizon_throttled)),
                long_horizon_throttle_reason=str(
                    event.get("throttle_reason", active_task.long_horizon_throttle_reason)
                ),
                running_stage=stage,
                updated_at=utc_now(),
            )
        elif stage == "pipeline.long_horizon_advisory_planned":
            active_task = replace(
                active_task,
                long_horizon_session_id=str(event.get("session_id", active_task.long_horizon_session_id)),
                running_stage=stage,
                updated_at=utc_now(),
            )
        elif stage == "pipeline.long_horizon_throttled":
            active_task = replace(
                active_task,
                long_horizon_session_id=str(event.get("session_id", active_task.long_horizon_session_id)),
                long_horizon_throttled=True,
                long_horizon_throttle_reason=str(event.get("reason", active_task.long_horizon_throttle_reason)),
                updated_at=utc_now(),
            )
        elif stage == "pipeline.long_horizon_paused":
            active_task = replace(
                active_task,
                long_horizon_session_id=str(event.get("session_id", active_task.long_horizon_session_id)),
                long_horizon_status="paused",
                long_horizon_completed_cycles=int(
                    event.get("completed_cycles", active_task.long_horizon_completed_cycles)
                ),
                long_horizon_total_cycles=int(event.get("total_cycles", active_task.long_horizon_total_cycles)),
                long_horizon_resume_count=int(event.get("resume_count", active_task.long_horizon_resume_count)),
                long_horizon_pause_requested=False,
                long_horizon_cancel_requested=False,
                running_stage=stage,
                updated_at=utc_now(),
            )
        elif stage == "pipeline.long_horizon_cancelled":
            active_task = replace(
                active_task,
                long_horizon_session_id=str(event.get("session_id", active_task.long_horizon_session_id)),
                long_horizon_status="cancelled",
                long_horizon_completed_cycles=int(
                    event.get("completed_cycles", active_task.long_horizon_completed_cycles)
                ),
                long_horizon_total_cycles=int(event.get("total_cycles", active_task.long_horizon_total_cycles)),
                long_horizon_pause_requested=False,
                long_horizon_cancel_requested=False,
                running_stage=stage,
                updated_at=utc_now(),
            )
        elif stage == "pipeline.long_horizon_completed":
            active_task = replace(
                active_task,
                long_horizon_session_id=str(event.get("session_id", active_task.long_horizon_session_id)),
                long_horizon_status="completed",
                long_horizon_completed_cycles=int(
                    event.get("completed_cycles", active_task.long_horizon_completed_cycles)
                ),
                long_horizon_total_cycles=int(event.get("total_cycles", active_task.long_horizon_total_cycles)),
                long_horizon_resume_count=int(event.get("resume_count", active_task.long_horizon_resume_count)),
                long_horizon_pause_requested=False,
                long_horizon_cancel_requested=False,
                running_stage=stage,
                updated_at=utc_now(),
            )
        elif stage == "pipeline.long_horizon_early_stopped":
            active_task = replace(
                active_task,
                long_horizon_session_id=str(event.get("session_id", active_task.long_horizon_session_id)),
                long_horizon_status="completed",
                long_horizon_current_phase="early_stopped",
                running_stage=stage,
                updated_at=utc_now(),
            )
        elif stage == "pipeline.long_horizon_failed":
            active_task = replace(
                active_task,
                long_horizon_session_id=str(event.get("session_id", active_task.long_horizon_session_id)),
                long_horizon_status="failed",
                running_stage=stage,
                updated_at=utc_now(),
            )
        elif stage in {
            "pipeline.planner_started",
            "pipeline.reasoner_started",
            "pipeline.critic_started",
            "pipeline.compressor_started",
            "pipeline.critic_repair_started",
            "pipeline.reasoner_repair_started",
            "pipeline.completed",
            "pipeline.failed",
            "pipeline.cancelled",
        }:
            active_task = replace(active_task, running_stage=stage, updated_at=utc_now())
        if stage == "pipeline.reasoner_done":
            active_task = replace(
                active_task,
                running_stage=stage,
                candidate_trace_count=int(event.get("candidate_trace_count", 0)),
                selected_candidate_id=str(event.get("selected_candidate_id", "")),
                selected_strategy=str(event.get("selected_strategy", "")),
                selected_verifier=str(event.get("selected_verifier", "")),
                candidate_score=float(event.get("candidate_score", 0.0) or 0.0),
                degraded_reason=str(event.get("degraded_reason", "")),
                updated_at=utc_now(),
            )
        elif stage in {"pipeline.critic_done", "pipeline.critic_repair_done"}:
            active_task = replace(
                active_task,
                critique_result=str(event.get("critique_result", "")),
                degraded_reason=str(event.get("degraded_reason", active_task.degraded_reason)),
                repair_actions=tuple(str(item) for item in event.get("repair_actions", ())),
                failure_categories=tuple(str(item) for item in event.get("failure_categories", ())),
                candidate_score=float(event.get("candidate_score", active_task.candidate_score) or 0.0),
                updated_at=utc_now(),
            )
        elif stage == "pipeline.completed":
            active_task = replace(
                active_task,
                running_stage=stage,
                answer_text=str(event.get("answer_text", "")),
                citation_refs=tuple(str(item) for item in event.get("citation_refs", ())),
                candidate_trace_count=int(event.get("candidate_trace_count", active_task.candidate_trace_count)),
                selected_candidate_id=str(event.get("selected_candidate_id", active_task.selected_candidate_id)),
                selected_strategy=str(event.get("selected_strategy", active_task.selected_strategy)),
                selected_verifier=str(event.get("selected_verifier", active_task.selected_verifier)),
                candidate_score=float(event.get("candidate_score", active_task.candidate_score) or 0.0),
                critique_result=str(event.get("critique_result", active_task.critique_result)),
                degraded_reason=str(event.get("degraded_reason", active_task.degraded_reason)),
                failure_categories=tuple(str(item) for item in event.get("failure_categories", ())),
                supporting_evidence_ids=tuple(str(item) for item in event.get("supporting_evidence_ids", ())),
                specialist_roles_used=tuple(str(item) for item in event.get("specialist_roles_used", ())),
                specialist_role_explanations=tuple(
                    str(item) for item in event.get("specialist_role_explanations", ())
                ),
                advisor_summaries=tuple(str(item) for item in event.get("advisor_summaries", ())),
                warning_count=int(event.get("warning_count", 0)),
                updated_at=utc_now(),
            )
        elif stage == "runtime.health_snapshot":
            runtime_health = DashboardRuntimeHealth.from_dict(event)
        elif stage == "dashboard.settings_profiles_loaded":
            settings_profiles = tuple(
                coerce_user_settings_profile(item) for item in event.get("profiles", ())
            )
        elif stage == "dashboard.task_history_loaded":
            task_history = tuple(
                coerce_dashboard_task_history_entry(item) for item in event.get("history", ())
            )
        elif stage == "dashboard.task_detail_loaded":
            selected_task = coerce_dashboard_task_inspector(event.get("task", {}))
        elif stage == "dashboard.knowledge_library_loaded":
            knowledge_sources = tuple(
                coerce_dashboard_knowledge_source(item) for item in event.get("sources", ())
            )
        elif stage == "dashboard.readiness_loaded":
            readiness_report = coerce_dashboard_readiness_report(event.get("report", {}))
        elif stage == "dashboard.model_registry_loaded":
            model_registry_view = coerce_model_registry_view(event.get("model_registry_view", {}))
        elif stage == "dashboard.model_role_action_reported":
            model_role_action = coerce_model_role_action_report(event.get("model_role_action", {}))
        elif stage == "dashboard.examples_loaded":
            demo_pack_status = coerce_demo_pack_status(event.get("demo_pack_status", {}))
            sample_tasks = tuple(
                coerce_sample_task_definition(item) for item in event.get("sample_tasks", ())
            )
            if event.get("selected_sample_task"):
                selected_sample_task = coerce_sample_task_definition(event.get("selected_sample_task", {}))
        elif stage == "dashboard.sample_task_selected":
            selected_sample_task = coerce_sample_task_definition(event.get("sample_task", {}))
        elif stage == "dashboard.audio_input_loaded":
            audio_input = coerce_audio_transcription_result(event.get("audio_input", {}))
        elif stage == "dashboard.audio_transcript_imported":
            audio_input = coerce_audio_transcription_result(event.get("audio_input", {}))
            if self._question_var is not None:
                self._question_var.set(str(event.get("question_text", audio_input.normalized_question)))
        elif stage == "dashboard.audio_input_cleared":
            audio_input = AudioTranscriptionResult()
        elif stage == "dashboard.audio_output_loaded":
            audio_output = coerce_audio_synthesis_result(event.get("audio_output", {}))
        elif stage == "dashboard.audio_output_cleared":
            audio_output = AudioSynthesisResult()
        elif stage == "dashboard.translation_output_loaded":
            translation_output = coerce_text_translation_result(event.get("translation_output", {}))
        elif stage == "dashboard.translation_imported":
            translation_output = coerce_text_translation_result(event.get("translation_output", {}))
            if self._question_var is not None:
                self._question_var.set(str(event.get("question_text", translation_output.translated_text)))
        elif stage == "dashboard.translation_output_cleared":
            translation_output = TextTranslationResult()
        elif stage == "dashboard.code_output_loaded":
            code_output = coerce_code_specialist_result(event.get("code_output", {}))
        elif stage == "dashboard.code_output_cleared":
            code_output = CodeSpecialistResult()
        elif stage == "dashboard.notice":
            last_notice = str(event.get("message", ""))
            last_notice_severity = str(event.get("severity", "info"))

        if stage.startswith("runtime.") and stage != "runtime.health_snapshot" and "reason" in event:
            recent_conditions.insert(
                0,
                RuntimeCondition.from_event(
                    stage,
                    event,
                    timestamp=event.get("timestamp"),
                ),
            )
            recent_conditions = recent_conditions[:10]

        dropped_events = self._dropped_events
        if "dropped_events" in event:
            dropped_events = int(event["dropped_events"])

        self._app_state = replace(
            state,
            last_stage=stage,
            event_count=state.event_count + 1,
            dropped_events=dropped_events,
            active_task=active_task,
            runtime_health=runtime_health,
            statuses=statuses,
            recent_conditions=tuple(recent_conditions),
            settings_profiles=settings_profiles,
            task_history=task_history,
            selected_task=selected_task,
            knowledge_sources=knowledge_sources,
            readiness_report=readiness_report,
            model_registry_view=model_registry_view,
            model_role_action=model_role_action,
            demo_pack_status=demo_pack_status,
            sample_tasks=sample_tasks,
            selected_sample_task=selected_sample_task,
            audio_input=audio_input,
            audio_output=audio_output,
            translation_output=translation_output,
            code_output=code_output,
            last_notice=last_notice,
            last_notice_severity=last_notice_severity,
            updated_at=utc_now(),
        )

    def _render_panels(self) -> None:
        state = self._app_state
        task = state.active_task
        health = state.runtime_health

        self._set_text(
            self._answer_text,
            task.answer_text or "No completed answer yet.",
        )
        self._set_text(
            self._citations_text,
            "\n".join(task.citation_refs) if task.citation_refs else "No citations yet.",
        )
        self._set_text(
            self._critique_text,
            "\n".join(
                (
                    f"Result: {task.critique_result or '(pending)'}",
                    f"Degraded reason: {task.degraded_reason or '(none)'}",
                    f"Candidate score: {task.candidate_score:.3f}",
                    f"Repair actions: {', '.join(task.repair_actions) if task.repair_actions else '(none)'}",
                    f"Failure categories: {', '.join(task.failure_categories) if task.failure_categories else '(none)'}",
                    (
                        "Specialist roles: "
                        + (", ".join(task.specialist_roles_used) if task.specialist_roles_used else "(none)")
                    ),
                    (
                        "Advisor suggestions: "
                        + (", ".join(task.advisor_summaries) if task.advisor_summaries else "(none)")
                    ),
                )
            ),
        )
        self._set_text(
            self._evidence_text,
            "\n".join(
                (
                    f"Supporting evidence IDs: {', '.join(task.supporting_evidence_ids) if task.supporting_evidence_ids else '(none)'}",
                    f"Local result count: {task.local_result_count}",
                    f"Web result count: {task.web_result_count}",
                    f"Used web fallback: {'yes' if task.used_web_fallback else 'no'}",
                )
            ),
        )
        self._set_text(
            self._provenance_text,
            "\n".join(
                (
                    f"Task ID: {task.task_id or '(none)'}",
                    f"Mode: {task.execution_mode or 'interactive'}",
                    f"Requested minutes: {task.requested_thinking_minutes or task.thinking_minutes or 0}",
                    f"Stage: {task.running_stage or state.last_stage or '(none)'}",
                    f"Selected candidate: {task.selected_candidate_id or '(none)'}",
                    f"Strategy: {task.selected_strategy or '(none)'}",
                    f"Verifier: {task.selected_verifier or '(none)'}",
                    f"Candidate traces: {task.candidate_trace_count}",
                    (
                        "Role explanations: "
                        + (
                            " | ".join(task.specialist_role_explanations)
                            if task.specialist_role_explanations
                            else "(none)"
                        )
                    ),
                )
            ),
        )
        self._set_text(
            self._web_text,
            "\n".join(
                (
                    f"Query: {task.web_query or '(none)'}",
                    f"Fallback used: {'yes' if task.used_web_fallback else 'no'}",
                    "Source refs:",
                    *(
                        [f"- {item}" for item in task.web_source_refs]
                        if task.web_source_refs
                        else ["- (none)"]
                    ),
                )
            ),
        )
        self._set_text(
            self._status_text,
            "\n".join(
                (
                    f"{component}: {status.state.value} [{status.severity.value}] {status.message}"
                    for component, status in sorted(state.statuses.items())
                )
            )
            if state.statuses
            else "No status updates yet.",
        )
        self._set_text(
            self._health_text,
            "\n".join(
                (
                    f"Generation backend: {health.generation_backend or '(unknown)'}",
                    f"Embedding backend: {health.embedding_backend or '(unknown)'}",
                    f"Generation jobs: {health.active_generation_jobs}",
                    f"Embedding jobs: {health.active_embedding_jobs}",
                    f"Fallback active: {'yes' if health.fallback_active else 'no'}",
                    f"Fallback reason: {health.fallback_reason or '(none)'}",
                    f"RAM available/total: {self._format_gb(health.available_ram_gb)} / {self._format_gb(health.total_ram_gb)}",
                    f"VRAM gen/embed: {self._format_gb(health.generation_backend_vram_gb)} / {self._format_gb(health.embedding_backend_vram_gb)}",
                    f"Last model error: {health.last_error or '(none)'}",
                )
            ),
        )
        self._set_text(
            self._model_registry_text,
            self._format_model_registry_view(state.model_registry_view),
        )
        self._set_text(
            self._model_role_detail_text,
            self._format_model_role_detail(
                self._selected_model_role_value(),
                state.model_registry_view,
                state.model_role_action,
            ),
        )
        if self._model_role_status_var is not None and state.model_role_action.action:
            self._model_role_status_var.set(state.model_role_action.summary or "Role action completed.")
        self._set_text(
            self._long_horizon_text,
            self._format_long_horizon_progress(task),
        )
        self._set_text(
            self._conditions_text,
            "\n".join(
                (
                    f"{condition.stage}: {condition.component} -> {condition.reason}"
                    + (
                        f" | Recovery: {', '.join(condition.metadata.get('recovery_actions', ())) }"
                        if condition.metadata.get("recovery_actions")
                        else ""
                    )
                    for condition in state.recent_conditions
                )
            )
            if state.recent_conditions
            else "No runtime conditions surfaced yet.",
        )
        self._set_text(
            self._settings_profiles_text,
            "\n".join(
                f"{profile.profile_name}: mode={profile.reasoning.get('mode', 'auto')} "
                f"backend={profile.runtime.get('generation_backend', '(default)')} "
                f"desktop={'on' if profile.desktop.get('enabled') else 'off'} "
                f"cloud={profile.cloud.get('mode', 'auxiliary_only')}"
                for profile in state.settings_profiles
            )
            if state.settings_profiles
            else "No saved settings profiles yet.",
        )
        history_lines: list[str] = []
        for entry in state.task_history:
            history_lines.extend(
                [
                    f"{entry.task_id} | {entry.completed_at.isoformat()}",
                    f"Q: {entry.question}",
                    f"A: {entry.answer_preview or '(no answer text)'}",
                    (
                        f"Critique={entry.critique_result or '(pending)'} | "
                        f"Verifier={entry.selected_verifier or '(none)'} | "
                        f"Warnings={entry.warning_count} | "
                        f"Web={'yes' if entry.used_web_fallback else 'no'}"
                    ),
                    "",
                ]
            )
        self._set_text(
            self._history_list_text,
            "\n".join(history_lines) if history_lines else "No task history loaded yet.",
        )
        selected = state.selected_task
        self._set_text(
            self._history_detail_text,
            "\n".join(
                (
                    f"Task ID: {selected.task_id or '(none)'}",
                    f"Question: {selected.question or '(none)'}",
                    f"Answer: {selected.answer_text or '(none)'}",
                    f"Citations: {', '.join(selected.citation_refs) if selected.citation_refs else '(none)'}",
                    f"Critique: {selected.critique_result or '(none)'}",
                    f"Degraded reason: {selected.degraded_reason or '(none)'}",
                    f"Repair actions: {', '.join(selected.repair_actions) if selected.repair_actions else '(none)'}",
                    f"Failure categories: {', '.join(selected.failure_categories) if selected.failure_categories else '(none)'}",
                    f"Evidence IDs: {', '.join(selected.supporting_evidence_ids) if selected.supporting_evidence_ids else '(none)'}",
                    f"Candidate traces: {selected.candidate_trace_count}",
                    f"Strategy/verifier: {selected.selected_strategy or '(none)'} / {selected.selected_verifier or '(none)'}",
                    (
                        "Specialist roles: "
                        + (", ".join(selected.specialist_roles_used) if selected.specialist_roles_used else "(none)")
                    ),
                    (
                        "Role explanations: "
                        + (
                            " | ".join(selected.specialist_role_explanations)
                            if selected.specialist_role_explanations
                            else "(none)"
                        )
                    ),
                    (
                        "Advisor suggestions: "
                        + (", ".join(selected.advisor_summaries) if selected.advisor_summaries else "(none)")
                    ),
                    f"Warnings: {', '.join(selected.warnings) if selected.warnings else '(none)'}",
                    f"Trace debug export: {selected.trace_debug_export_path or '(not exported)'}",
                    "Optimizer lifecycle: "
                    + (", ".join(selected.optimizer_lifecycle) if selected.optimizer_lifecycle else "(none)"),
                )
            ),
        )
        knowledge_lines: list[str] = []
        for source in state.knowledge_sources:
            knowledge_lines.extend(
                [
                    f"{source.source_ref} | {source.title}",
                    f"Chunks={source.chunk_count} | Model={source.embedding_model or '(unknown)'} | "
                    f"Archived={'yes' if source.archived else 'no'} | "
                    f"Origin={source.corpus_origin or '(user)'} | Tier={source.corpus_tier or '(none)'}",
                    "",
                ]
            )
        self._set_text(
            self._knowledge_list_text,
            "\n".join(knowledge_lines) if knowledge_lines else "No knowledge sources loaded yet.",
        )
        examples = state.sample_tasks
        selected_sample = state.selected_sample_task
        examples_lines: list[str] = []
        for sample_task in examples:
            examples_lines.extend(
                [
                    f"{sample_task.sample_id} | {sample_task.title}",
                    f"Category={sample_task.category} | Profile={sample_task.execution_profile} | "
                    f"Minutes={sample_task.recommended_thinking_minutes} | "
                    f"Demo pack={'yes' if sample_task.requires_demo_pack else 'no'} | "
                    f"Web={'yes' if sample_task.uses_web_fallback else 'no'}",
                    f"Expected: {sample_task.expected_behavior or '(none)'}",
                    "",
                ]
            )
        self._set_text(
            self._examples_list_text,
            "\n".join(examples_lines) if examples_lines else "No Phase 11 sample tasks loaded yet.",
        )
        demo_pack = state.demo_pack_status
        self._set_text(
            self._examples_detail_text,
            "\n".join(
                (
                    f"Pack version: {demo_pack.pack_version or '(unknown)'}",
                    f"Demo pack loaded: {'yes' if demo_pack.loaded else 'no'}",
                    f"Loaded documents: {demo_pack.loaded_document_count}/{demo_pack.document_count}",
                    (
                        "Runtime pack: "
                        f"macros {demo_pack.runtime_pack.loaded_macro_count}/{len(demo_pack.runtime_pack.macro_names)}, "
                        f"opcodes {demo_pack.runtime_pack.loaded_opcode_count}/{len(demo_pack.runtime_pack.opcode_names)}, "
                        f"decoders {demo_pack.runtime_pack.loaded_decoder_count}/{len(demo_pack.runtime_pack.decoder_names)}"
                    ),
                    f"Verified trace fixture: {demo_pack.verified_trace_example_path or '(none)'}",
                    f"Status: {demo_pack.status_detail or '(none)'}",
                    "",
                    f"Selected sample: {selected_sample.title or '(none)'}",
                    f"Sample ID: {selected_sample.sample_id or '(none)'}",
                    f"Question: {selected_sample.question or '(none)'}",
                    f"Expected result: {selected_sample.expected_result or '(not fixed)'}",
                    (
                        "Success markers: "
                        + (", ".join(selected_sample.success_markers) if selected_sample.success_markers else "(none)")
                    ),
                    (
                        "Required sources: "
                        + (
                            ", ".join(selected_sample.required_source_refs)
                            if selected_sample.required_source_refs
                            else "(none)"
                        )
                    ),
                    (
                        "Fast/deep comparison: "
                        + (
                            f"{selected_sample.comparison_fast_minutes}m vs {selected_sample.comparison_deep_minutes}m"
                            if selected_sample.comparison_group
                            else "(none)"
                        )
                    ),
                    f"Expected degraded reason: {selected_sample.expected_degraded_reason or '(none)'}",
                )
            ),
        )
        audio_input = state.audio_input
        audio_output = state.audio_output
        translation_output = state.translation_output
        code_output = state.code_output
        audio_lines = [
            "Voice input:",
            f"Status: {audio_input.status}",
            f"Source path: {audio_input.source_path or '(none)'}",
            (
                "Backend/model: "
                f"{audio_input.transcription_backend or '(none)'} / "
                f"{audio_input.transcription_model or '(none)'}"
            ),
            (
                "Duration analyzed: "
                f"{audio_input.analyzed_duration_seconds:.2f}s / {audio_input.duration_seconds:.2f}s"
            ),
            f"Used VAD: {'yes' if audio_input.used_vad else 'no'}",
            (
                "Speech segments: "
                f"{audio_input.voice_activity.segment_count} | "
                f"ratio {audio_input.voice_activity.speech_ratio:.2f}"
            ),
            f"Transcript: {audio_input.transcript_text or '(none)'}",
            f"Question import: {audio_input.normalized_question or '(none)'}",
            (
                "Warnings: "
                + (", ".join(audio_input.warnings) if audio_input.warnings else "(none)")
            ),
            "",
            "Voice output:",
            f"Status: {audio_output.status}",
            f"Target path: {audio_output.target_path or '(none)'}",
            (
                "Backend/model: "
                f"{audio_output.synthesis_backend or '(none)'} / "
                f"{audio_output.synthesis_model or '(none)'}"
            ),
            f"Duration/sample rate: {audio_output.duration_seconds:.2f}s / {audio_output.sample_rate_hz or 0} Hz",
            f"Source text: {audio_output.source_text or '(none)'}",
            f"Spoken text: {audio_output.clipped_text or '(none)'}",
            (
                "Warnings: "
                + (", ".join(audio_output.warnings) if audio_output.warnings else "(none)")
            ),
        ]
        self._set_text(
            self._audio_text,
            "\n".join(audio_lines)
            if (
                audio_input.source_path
                or audio_input.transcript_text
                or audio_output.target_path
                or audio_output.source_text
            )
            else "No audio input or output loaded yet.",
        )
        translation_lines = [
            f"Status: {translation_output.status}",
            f"Source/target: {translation_output.source_language or '(auto)'} -> {translation_output.target_language or '(none)'}",
            (
                "Backend/model: "
                f"{translation_output.translation_backend or '(none)'} / "
                f"{translation_output.translation_model or '(none)'}"
            ),
            f"Scope: {translation_output.source_scope or '(none)'}",
            f"Source text: {translation_output.source_text or '(none)'}",
            f"Translated text: {translation_output.translated_text or '(none)'}",
            (
                "Warnings: "
                + (", ".join(translation_output.warnings) if translation_output.warnings else "(none)")
            ),
        ]
        self._set_text(
            self._translation_output_text,
            "\n".join(translation_lines)
            if translation_output.source_text or translation_output.translated_text
            else "No translation result loaded yet.",
        )
        code_lines = [
            f"Status: {code_output.status}",
            f"Scope/path: {code_output.source_scope or '(none)'} / {code_output.source_path or '(none)'}",
            (
                "Backend/model: "
                f"{code_output.code_backend or '(none)'} / "
                f"{code_output.code_model or '(none)'}"
            ),
            f"Language: {code_output.detected_language or '(unknown)'}",
            f"Line count: {code_output.line_count}",
            f"Request: {code_output.request_text or '(none)'}",
            f"Summary: {code_output.summary or '(none)'}",
            (
                "Suggested actions: "
                + (", ".join(code_output.suggested_actions) if code_output.suggested_actions else "(none)")
            ),
            (
                "Warnings: "
                + (", ".join(code_output.warnings) if code_output.warnings else "(none)")
            ),
        ]
        self._set_text(
            self._code_output_text,
            "\n".join(code_lines)
            if code_output.summary or code_output.source_path or code_output.request_text
            else "No code-specialist result loaded yet.",
        )
        readiness = state.readiness_report
        check_lines: list[str] = [
            f"Stub mode ready: {'yes' if readiness.stub_mode_ready else 'no'}",
            f"Real mode ready: {'yes' if readiness.real_mode_ready else 'no'}",
            "",
        ]
        for check in readiness.checks:
            check_lines.append(f"{check.title}: {check.status}")
            if check.detail:
                check_lines.append(check.detail)
            if check.recovery_actions:
                check_lines.append("Recovery: " + ", ".join(check.recovery_actions))
            check_lines.append("")
        if readiness.guidance:
            check_lines.append("Guidance:")
            check_lines.extend(f"- {item}" for item in readiness.guidance)
        self._set_text(
            self._readiness_checks_text,
            "\n".join(check_lines).strip() if readiness.checks or readiness.guidance else "No readiness report loaded yet.",
        )
        capability_lines: list[str] = []
        for capability in readiness.capabilities:
            capability_lines.extend(
                [
                    f"{capability.capability_name}: {capability.status}",
                    f"Reason: {capability.reason or '(none)'}",
                    f"Detail: {capability.detail or '(none)'}",
                    (
                        "Recovery: " + ", ".join(capability.recovery_actions)
                        if capability.recovery_actions
                        else "Recovery: (none)"
                    ),
                    "",
                ]
            )
        self._set_text(
            self._capability_text,
            "\n".join(capability_lines) if capability_lines else "No capability-gating data yet.",
        )
        self._set_debug_pane_visibility(bool(state.user_settings.ui.get("show_debug_pane", True)))
        if self._app_notice_var is not None:
            self._app_notice_var.set(state.last_notice or "Local app shell ready.")
        if self._knowledge_status_var is not None and state.last_notice:
            self._knowledge_status_var.set(state.last_notice)
        if self._examples_status_var is not None:
            self._examples_status_var.set(
                demo_pack.status_detail
                or state.last_notice
                or "Phase 11 demo pack available."
            )
        if self._audio_status_var is not None:
            if audio_output.target_path or audio_output.source_text:
                self._audio_status_var.set(
                    f"Voice output {audio_output.status}: {audio_output.target_path or '(none)'}"
                )
            elif audio_input.source_path or audio_input.transcript_text:
                self._audio_status_var.set(
                    f"Voice input {audio_input.status}: {audio_input.source_path or '(none)'}"
                )
            else:
                self._audio_status_var.set("Voice input and output ready.")
        if self._translation_status_var is not None:
            if translation_output.translated_text:
                self._translation_status_var.set(
                    f"Translation {translation_output.status}: {translation_output.source_language or '(auto)'} -> {translation_output.target_language or '(none)'}"
                )
            else:
                self._translation_status_var.set("Translation role ready.")
        if self._code_status_var is not None:
            if code_output.summary or code_output.source_path:
                self._code_status_var.set(
                    f"Code specialist {code_output.status}: {code_output.source_path or code_output.source_scope}"
                )
            else:
                self._code_status_var.set("Code specialist ready.")
        current_sample_id = self._sample_task_id_var.get() if self._sample_task_id_var is not None else ""
        if self._sample_task_id_var is not None and selected_sample.sample_id:
            if current_sample_id != selected_sample.sample_id:
                self._sample_task_id_var.set(selected_sample.sample_id)
                if self._question_var is not None and selected_sample.question:
                    self._question_var.set(selected_sample.question)
                if self._thinking_minutes_var is not None and selected_sample.recommended_thinking_minutes > 0:
                    self._thinking_minutes_var.set(selected_sample.recommended_thinking_minutes)
        if self._thinking_label_var is not None and self._thinking_minutes_var is not None:
            self._thinking_label_var.set(f"{int(self._thinking_minutes_var.get())} minutes")
        self._refresh_time_control_summary()

    def _set_text(self, widget: tk.Text | None, content: str) -> None:
        if widget is None:
            return
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", content)
        widget.configure(state="disabled")

    def _safe_int(self, value: Any, default: int) -> int:
        try:
            if isinstance(value, str):
                value = value.strip() or default
            return int(value)
        except (TypeError, ValueError):
            return default

    def _planned_submission_minutes(self) -> int:
        thinking_minutes = int(self._thinking_minutes_var.get()) if self._thinking_minutes_var is not None else 30
        if self._long_horizon_enabled_var is None or not bool(self._long_horizon_enabled_var.get()):
            return thinking_minutes
        wall_clock_minutes = self._safe_int(
            self._long_horizon_minutes_var.get() if self._long_horizon_minutes_var is not None else 120,
            120,
        )
        return max(thinking_minutes, wall_clock_minutes)

    def _planned_cycle_budget_minutes(self, planned_minutes: int) -> int:
        if planned_minutes <= 120:
            return planned_minutes
        current_cycle_budget = self._app_state.user_settings.long_horizon.get("cycle_budget_minutes", 120)
        return max(1, min(self._safe_int(current_cycle_budget, 120), planned_minutes))

    def _planned_execution_mode(self) -> str:
        return "long_horizon" if self._planned_submission_minutes() > 120 else "interactive"

    def _active_long_horizon_session_id(self) -> str:
        return self._app_state.active_task.long_horizon_session_id.strip()

    def _refresh_time_control_summary(self) -> None:
        if self._time_summary_var is None:
            return
        task = self._app_state.active_task
        if task.execution_mode == "long_horizon" and task.long_horizon_session_id:
            summary = (
                "Active mode: Long-horizon"
                f" | Wall clock {task.requested_thinking_minutes or task.thinking_minutes} min"
                f" | Cycle {task.long_horizon_cycle_budget_minutes or '(n/a)'} min"
                f" | Checkpoints {task.long_horizon_completed_cycles}/{task.long_horizon_total_cycles}"
                f" | Phase {self._format_phase_label(task.long_horizon_current_phase or task.long_horizon_status)}"
            )
        else:
            planned_minutes = self._planned_submission_minutes()
            planned_mode = self._planned_execution_mode()
            if planned_mode == "long_horizon":
                cycle_budget = self._planned_cycle_budget_minutes(planned_minutes)
                summary = (
                    "Planned mode: Long-horizon"
                    f" | Wall clock {planned_minutes} min"
                    f" | Cycle {cycle_budget} min"
                    " | Checkpointed and resumable"
                )
            else:
                summary = f"Planned mode: Interactive | {planned_minutes} minute budget."
        self._time_summary_var.set(summary)

    def _format_duration(self, seconds: float | None) -> str:
        if seconds is None:
            return "(unknown)"
        total_seconds = max(0, int(round(seconds)))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        parts: list[str] = []
        if hours:
            parts.append(f"{hours}h")
        if minutes or hours:
            parts.append(f"{minutes}m")
        parts.append(f"{secs}s")
        return " ".join(parts)

    def _format_phase_label(self, phase: str) -> str:
        normalized = phase.strip().replace("_", " ")
        return normalized if normalized else "(none)"

    def _format_model_registry_view(self, view: ModelRegistryView) -> str:
        registrations = view.registrations
        if not registrations:
            return "No local-AI registry view published yet."
        role_lines: list[str] = []
        registration_lines: list[str] = []
        for role in ModelRole:
            role_registrations = tuple(item for item in registrations if item.role == role)
            preferred_id = view.preferred_models.get(role.value, "")
            enabled_count = sum(1 for item in role_registrations if item.enabled and item.backend != "unconfigured")
            recent_fallbacks = []
            for decision in reversed(view.last_route_decisions):
                if decision.requested_role != role or not decision.fallback_reason:
                    continue
                if decision.fallback_reason not in recent_fallbacks:
                    recent_fallbacks.append(decision.fallback_reason)
                if len(recent_fallbacks) >= 2:
                    break
            fallback_label = ", ".join(recent_fallbacks) if recent_fallbacks else view.fallback_reasons.get(role.value, "(none)")
            role_lines.append(
                f"{role.value}: {enabled_count}/{len(role_registrations)} enabled"
                + (f" | preferred {preferred_id}" if preferred_id else "")
                + f" | fallback {fallback_label}"
            )
            for registration in role_registrations:
                markers: list[str] = []
                if registration.registration_id == preferred_id:
                    markers.append("preferred")
                if registration.role.value in view.active_heavy_roles and registration.resource_class.value == "heavy":
                    markers.append("active")
                if registration.backend == "unconfigured":
                    markers.append("placeholder")
                registration_lines.append(
                    f"{registration.role.value}: {registration.backend} / {registration.model_identifier}"
                    f" | {'enabled' if registration.enabled else 'disabled'}"
                    f" | {registration.resource_class.value}"
                    + (f" | {', '.join(markers)}" if markers else "")
                    + (
                        f" | missing {', '.join(registration.missing_dependencies)}"
                        if registration.missing_dependencies
                        else ""
                    )
                )
        route_lines = [
            (
                f"{decision.requested_role.value}: "
                f"{decision.selected_backend or '(none)'} / {decision.selected_model_identifier or '(none)'}"
                f" | {'allowed' if decision.allowed else 'blocked'}"
                + (f" | {decision.fallback_reason}" if decision.fallback_reason else "")
            )
            for decision in view.last_route_decisions[-4:]
        ]
        cache_lines = [
            (
                f"{snapshot.namespace}: {snapshot.entry_count}/{snapshot.max_entries}"
                f" | hits {snapshot.hits} | misses {snapshot.misses} | evictions {snapshot.evictions}"
            )
            for snapshot in view.cache_snapshots
        ]
        suggestion_lines = [
            (
                f"{record.kind.value}: {record.summary}"
                f" | conf {record.confidence:.2f}"
                + (
                    f" | targets {', '.join(record.target_components)}"
                    if record.target_components
                    else ""
                )
            )
            for record in view.recent_optimizer_suggestions
        ]
        compression_lines = [
            (
                f"{item.macro_name or item.proposal_id}: gain {item.estimated_gain:.2f}"
                f" | validation {item.validation_state}"
                f" | pass {item.validation_pass_rate:.2f}"
                + (f" | blocked {item.blocked_reason}" if item.blocked_reason else "")
                + (f" | basis {item.evidence_basis}" if item.evidence_basis else "")
            )
            for item in view.compression_insights
        ]
        return "\n".join(
            (
                f"Heavy slots: {len(view.active_heavy_roles)}/{view.heavy_slot_limit}",
                (
                    "Active heavy roles: "
                    + (", ".join(view.active_heavy_roles) if view.active_heavy_roles else "(none)")
                ),
                (
                    "Advisory available: "
                    + ("yes" if view.advisory_available else "no")
                    + (
                        f" | subscriptions {', '.join(view.optimizer_subscriptions)}"
                        if view.optimizer_subscriptions
                        else ""
                    )
                ),
                (
                    "Preferred models: "
                    + (
                        ", ".join(f"{role} -> {value}" for role, value in sorted(view.preferred_models.items()))
                        if view.preferred_models
                        else "(none)"
                    )
                ),
                (
                    "Fallback reasons: "
                    + (
                        ", ".join(f"{role} -> {reason}" for role, reason in sorted(view.fallback_reasons.items()))
                        if view.fallback_reasons
                        else "(none)"
                    )
                ),
                "Role summary:",
                *role_lines,
                "Registrations:",
                *registration_lines,
                "Recent route decisions:",
                *(route_lines or ["(none)"]),
                "Recent optimizer suggestions:",
                *(suggestion_lines or ["(none)"]),
                "Compression insights:",
                *(compression_lines or ["(none)"]),
                "Bounded caches:",
                *(cache_lines or ["(none)"]),
            )
        )

    def _format_model_role_detail(
        self,
        role_value: str,
        view: ModelRegistryView,
        action_report: ModelRoleActionReport,
    ) -> str:
        try:
            role = ModelRole(role_value)
        except ValueError:
            return "Select a valid local-AI role to inspect."
        registrations = tuple(item for item in view.registrations if item.role == role)
        preferred_id = view.preferred_models.get(role.value, "")
        route_decisions = tuple(
            decision for decision in view.last_route_decisions if decision.requested_role == role
        )[-4:]
        fallback_reasons: list[str] = []
        if role.value in view.fallback_reasons:
            fallback_reasons.append(view.fallback_reasons[role.value])
        for decision in reversed(route_decisions):
            if decision.fallback_reason and decision.fallback_reason not in fallback_reasons:
                fallback_reasons.append(decision.fallback_reason)
        lines = [
            f"Role: {role.value}",
            f"Preferred registration: {preferred_id or '(none)'}",
            (
                "Current heavy residency: active"
                if role.value in view.active_heavy_roles
                else "Current heavy residency: inactive"
            ),
            "Registrations:",
            *(
                [
                    (
                        f"- {registration.backend} / {registration.model_identifier}"
                        f" | {'enabled' if registration.enabled else 'disabled'}"
                        f" | {registration.resource_class.value}"
                    )
                    for registration in registrations
                ]
                or ["- (none)"]
            ),
            "Recent route decisions:",
            *(
                [
                    (
                        f"- {decision.capability or 'route'} -> "
                        f"{decision.selected_backend or '(none)'} / {decision.selected_model_identifier or '(none)'}"
                        f" | {'allowed' if decision.allowed else 'blocked'}"
                        + (f" | {decision.fallback_reason}" if decision.fallback_reason else "")
                    )
                    for decision in route_decisions
                ]
                or ["- (none)"]
            ),
            "Fallback reasons:",
            *([f"- {reason}" for reason in fallback_reasons] or ["- (none)"]),
            "Optimizer suggestions:",
            *(
                [
                    f"- {record.kind.value}: {record.summary}"
                    for record in view.recent_optimizer_suggestions[:4]
                ]
                or ["- (none)"]
            ),
        ]
        if action_report.role == role and action_report.action:
            lines.extend(
                [
                    "Last quick action:",
                    f"- {action_report.action}: {action_report.summary}",
                    f"- Detail: {action_report.detail or '(none)'}",
                ]
            )
            if action_report.guidance:
                lines.extend(["Guidance:", *[f"- {item}" for item in action_report.guidance]])
        return "\n".join(lines)

    def _format_long_horizon_progress(self, task) -> str:
        if task.execution_mode != "long_horizon":
            planned_minutes = self._planned_submission_minutes()
            if self._planned_execution_mode() == "long_horizon":
                return "\n".join(
                    (
                        "Mode: Long-horizon (planned)",
                        f"Wall-clock budget: {planned_minutes} minutes",
                        f"Cycle budget: {self._planned_cycle_budget_minutes(planned_minutes)} minutes",
                        "Status: waiting to start",
                        "Extra-time summary appears here after the first checkpoint.",
                    )
                )
            return "\n".join(
                (
                    "Mode: Interactive",
                    f"Current budget: {planned_minutes} minutes",
                    "Use presets above 120 minutes to enter checkpointed long-horizon mode.",
                )
            )

        supporting_evidence_count = len(task.supporting_evidence_ids)
        first_critique = task.long_horizon_first_critique_result or "(none)"
        latest_critique = task.critique_result or "(pending)"
        return "\n".join(
            (
                "Mode: Long-horizon",
                f"Session: {task.long_horizon_session_id or '(none)'}",
                (
                    "Status / phase: "
                    f"{task.long_horizon_status or '(pending)'} / "
                    f"{self._format_phase_label(task.long_horizon_current_phase)}"
                ),
                (
                    "Budget: "
                    f"{task.requested_thinking_minutes or task.thinking_minutes} min wall clock"
                    f" | cycle {task.long_horizon_cycle_budget_minutes or '(n/a)'} min"
                    f" | checkpoint {task.long_horizon_checkpoint_interval_minutes or '(n/a)'} min"
                ),
                (
                    "Progress: "
                    f"{task.long_horizon_completed_cycles}/{task.long_horizon_total_cycles} checkpoints"
                    f" | elapsed {self._format_duration(task.long_horizon_elapsed_seconds)}"
                    f" | budget ETA {self._format_duration(task.long_horizon_eta_seconds)}"
                ),
                (
                    "Control: "
                    f"resume_count={task.long_horizon_resume_count}"
                    f" | pause_requested={'yes' if task.long_horizon_pause_requested else 'no'}"
                    f" | cancel_requested={'yes' if task.long_horizon_cancel_requested else 'no'}"
                ),
                (
                    "Throttling: "
                    f"{'yes' if task.long_horizon_throttled else 'no'}"
                    + (
                        f" | reason {task.long_horizon_throttle_reason}"
                        if task.long_horizon_throttle_reason
                        else ""
                    )
                ),
                "",
                "What extra time bought:",
                (
                    "Candidate depth: "
                    f"start {task.long_horizon_initial_candidate_count}"
                    f" | current {task.candidate_trace_count}"
                    f" | peak {task.long_horizon_peak_candidate_count}"
                    f" | +{task.long_horizon_additional_candidate_count}"
                ),
                (
                    "Supporting evidence: "
                    f"start {task.long_horizon_initial_supporting_evidence_count}"
                    f" | current {supporting_evidence_count}"
                    f" | +{task.long_horizon_additional_supporting_evidence_count}"
                ),
                f"Verification passes completed: {task.long_horizon_total_verification_passes}",
                f"Repairs applied: {task.long_horizon_total_repairs}",
                (
                    "Confidence gain: "
                    f"{task.long_horizon_confidence_gain:+.3f}"
                    f" (start {task.long_horizon_first_candidate_score:.3f}"
                    f" -> current {task.candidate_score:.3f})"
                ),
                (
                    "Validity improvement: "
                    f"{'yes' if task.long_horizon_validity_improved else 'no'}"
                    f" ({first_critique} -> {latest_critique})"
                ),
                "",
                "Advisor summary:",
                (
                    "Requested / accepted / rejected / deferred: "
                    f"{task.long_horizon_advisory_requested_count}"
                    f" / {task.long_horizon_advisory_accepted_count}"
                    f" / {task.long_horizon_advisory_rejected_count}"
                    f" / {task.long_horizon_advisory_deferred_count}"
                ),
                *(
                    [f"- {item}" for item in task.long_horizon_advisory_entries]
                    if task.long_horizon_advisory_entries
                    else ["- (no advisory suggestions yet)"]
                ),
                (
                    f"Early stop reason: {task.long_horizon_early_stop_reason}"
                    if task.long_horizon_early_stop_reason
                    else "Early stop reason: (none)"
                ),
            )
        )

    def _merge_long_horizon_event_metrics(self, task, event: dict[str, Any]):
        session_id = str(event.get("session_id", task.long_horizon_session_id))
        if not session_id:
            return task
        raw_eta_seconds = event.get("eta_seconds", task.long_horizon_eta_seconds)
        candidate_count = int(event.get("candidate_trace_count", task.candidate_trace_count))
        candidate_score = float(event.get("candidate_score", task.candidate_score) or 0.0)
        critique_result = str(event.get("critique_result", task.critique_result))
        repair_actions = tuple(str(item) for item in event.get("repair_actions", task.repair_actions))
        supporting_evidence_ids = tuple(str(item) for item in event.get("supporting_evidence_ids", task.supporting_evidence_ids))
        return replace(
            task,
            execution_mode=str(event.get("execution_mode", task.execution_mode or "long_horizon")),
            requested_thinking_minutes=int(
                event.get("requested_minutes", task.requested_thinking_minutes or task.thinking_minutes)
            ),
            thinking_minutes=int(event.get("requested_minutes", task.thinking_minutes or 0)),
            candidate_trace_count=candidate_count,
            selected_candidate_id=str(event.get("selected_candidate_id", task.selected_candidate_id)),
            candidate_score=candidate_score,
            critique_result=critique_result,
            repair_actions=repair_actions,
            supporting_evidence_ids=supporting_evidence_ids,
            local_result_count=int(event.get("local_result_count", task.local_result_count)),
            web_result_count=int(event.get("web_result_count", task.web_result_count)),
            used_web_fallback=bool(event.get("used_web_fallback", task.used_web_fallback)),
            long_horizon_session_id=session_id,
            long_horizon_current_phase=str(event.get("current_phase", task.long_horizon_current_phase)),
            long_horizon_cycle_budget_minutes=int(
                event.get("cycle_budget_minutes", task.long_horizon_cycle_budget_minutes)
            ),
            long_horizon_checkpoint_interval_minutes=int(
                event.get(
                    "checkpoint_interval_minutes",
                    task.long_horizon_checkpoint_interval_minutes,
                )
            ),
            long_horizon_duty_cycle_ratio=float(
                event.get("duty_cycle_ratio", task.long_horizon_duty_cycle_ratio) or 0.0
            ),
            long_horizon_cooldown_seconds=float(
                event.get("cooldown_seconds", task.long_horizon_cooldown_seconds) or 0.0
            ),
            long_horizon_elapsed_seconds=float(
                event.get("elapsed_seconds", task.long_horizon_elapsed_seconds) or 0.0
            ),
            long_horizon_eta_seconds=(
                task.long_horizon_eta_seconds
                if raw_eta_seconds in (None, "")
                else float(raw_eta_seconds)
            ),
            long_horizon_initial_candidate_count=int(
                event.get("initial_candidate_count", task.long_horizon_initial_candidate_count or candidate_count)
            ),
            long_horizon_peak_candidate_count=int(
                event.get("peak_candidate_count", task.long_horizon_peak_candidate_count or candidate_count)
            ),
            long_horizon_additional_candidate_count=int(
                event.get("additional_candidate_count", task.long_horizon_additional_candidate_count)
            ),
            long_horizon_initial_supporting_evidence_count=int(
                event.get(
                    "initial_supporting_evidence_count",
                    task.long_horizon_initial_supporting_evidence_count or len(supporting_evidence_ids),
                )
            ),
            long_horizon_additional_supporting_evidence_count=int(
                event.get(
                    "additional_supporting_evidence_count",
                    task.long_horizon_additional_supporting_evidence_count,
                )
            ),
            long_horizon_total_verification_passes=int(
                event.get("total_verification_passes", task.long_horizon_total_verification_passes)
            ),
            long_horizon_total_repairs=int(
                event.get("total_repairs", task.long_horizon_total_repairs)
            ),
            long_horizon_first_candidate_score=float(
                event.get("first_candidate_score", task.long_horizon_first_candidate_score or candidate_score) or 0.0
            ),
            long_horizon_confidence_gain=float(
                event.get("confidence_gain", task.long_horizon_confidence_gain) or 0.0
            ),
            long_horizon_first_critique_result=str(
                event.get("first_critique_result", task.long_horizon_first_critique_result or critique_result)
            ),
            long_horizon_validity_improved=bool(
                event.get("validity_improved", task.long_horizon_validity_improved)
            ),
            long_horizon_advisory_requested_count=int(
                event.get("advisory_requested_count", task.long_horizon_advisory_requested_count)
            ),
            long_horizon_advisory_accepted_count=int(
                event.get("advisory_accepted_count", task.long_horizon_advisory_accepted_count)
            ),
            long_horizon_advisory_rejected_count=int(
                event.get("advisory_rejected_count", task.long_horizon_advisory_rejected_count)
            ),
            long_horizon_advisory_deferred_count=int(
                event.get("advisory_deferred_count", task.long_horizon_advisory_deferred_count)
            ),
            long_horizon_advisory_entries=tuple(
                str(item)
                for item in event.get("advisory_entries", task.long_horizon_advisory_entries)
            ),
            long_horizon_early_stop_reason=str(
                event.get("early_stop_reason", task.long_horizon_early_stop_reason)
            ),
        )

    def _apply_settings_to_form(self, profile: UserSettingsProfile) -> None:
        if self._profile_name_var is not None:
            self._profile_name_var.set(profile.profile_name)
        if self._generation_backend_var is not None:
            self._generation_backend_var.set(str(profile.runtime.get("generation_backend", "ollama")))
        if self._embedding_backend_var is not None:
            self._embedding_backend_var.set(str(profile.runtime.get("embedding_backend", "sentence_transformers")))
        if self._vector_store_var is not None:
            self._vector_store_var.set(str(profile.runtime.get("vector_store_backend", "chromadb")))
        if self._web_provider_var is not None:
            self._web_provider_var.set(str(profile.retrieval.get("provider", "wikipedia")))
        if self._reasoning_mode_var is not None:
            self._reasoning_mode_var.set(str(profile.reasoning.get("mode", "auto")))
        if self._thinking_minutes_var is not None:
            self._thinking_minutes_var.set(int(profile.reasoning.get("thinking_minutes", 30) or 30))
        if self._thinking_label_var is not None and self._thinking_minutes_var is not None:
            self._thinking_label_var.set(f"{int(self._thinking_minutes_var.get())} minutes")
        if self._allow_web_fallback_var is not None:
            self._allow_web_fallback_var.set(bool(profile.retrieval.get("allow_web_fallback", True)))
        if self._enable_self_optimizer_var is not None:
            self._enable_self_optimizer_var.set(bool(profile.runtime.get("enable_self_optimizer", False)))
        if self._reranking_var is not None:
            self._reranking_var.set(bool(profile.retrieval.get("reranking", True)))
        enabled_roles = {
            str(item)
            for item in profile.models.get("enabled_roles", ())
            if str(item).strip()
        }
        if self._reranker_role_enabled_var is not None:
            self._reranker_role_enabled_var.set("reranker" in enabled_roles)
        if self._speech_to_text_role_enabled_var is not None:
            self._speech_to_text_role_enabled_var.set("speech_to_text" in enabled_roles)
        if self._text_to_speech_role_enabled_var is not None:
            self._text_to_speech_role_enabled_var.set("text_to_speech" in enabled_roles)
        if self._vad_role_enabled_var is not None:
            self._vad_role_enabled_var.set("vad" in enabled_roles)
        if self._translation_role_enabled_var is not None:
            self._translation_role_enabled_var.set("translation" in enabled_roles)
        if self._code_specialist_role_enabled_var is not None:
            self._code_specialist_role_enabled_var.set("code_specialist" in enabled_roles)
        if self._long_horizon_enabled_var is not None:
            self._long_horizon_enabled_var.set(bool(profile.long_horizon.get("enabled", False)))
        if self._long_horizon_minutes_var is not None:
            self._long_horizon_minutes_var.set(str(profile.long_horizon.get("wall_clock_minutes", 120)))
        if self._optimizer_policy_var is not None:
            self._optimizer_policy_var.set(str(profile.optimizer.get("activation_policy", "proposal_only")))
        if self._optimizer_replay_limit_var is not None:
            self._optimizer_replay_limit_var.set(str(profile.optimizer.get("replay_limit", 64)))
        if self._show_debug_pane_var is not None:
            self._show_debug_pane_var.set(bool(profile.ui.get("show_debug_pane", True)))
        if self._desktop_enabled_var is not None:
            self._desktop_enabled_var.set(bool(profile.desktop.get("enabled", False)))
        if self._desktop_approval_policy_var is not None:
            self._desktop_approval_policy_var.set(str(profile.desktop.get("approval_policy", "approve_risky_only")))
        if self._observation_tier_var is not None:
            self._observation_tier_var.set(str(profile.observation.get("tier", "screenshot_on_demand")))
        if self._cloud_mode_var is not None:
            self._cloud_mode_var.set(str(profile.cloud.get("mode", "auxiliary_only")))
        if self._log_runtime_events_var is not None:
            self._log_runtime_events_var.set(bool(profile.privacy.get("log_runtime_events", True)))
        if self._allow_cloud_content_var is not None:
            self._allow_cloud_content_var.set(bool(profile.privacy.get("allow_cloud_content", False)))
        if self._log_level_var is not None:
            self._log_level_var.set(str(profile.privacy.get("log_level", "INFO")))
        self._set_debug_pane_visibility(bool(profile.ui.get("show_debug_pane", True)))
        self._refresh_time_control_summary()

    def _gather_settings_from_form(self) -> UserSettingsProfile:
        current = self._app_state.user_settings
        profile_name = (self._profile_name_var.get().strip() if self._profile_name_var is not None else "") or "default"
        thinking_minutes = int(self._thinking_minutes_var.get()) if self._thinking_minutes_var is not None else 30
        reasoning_mode = self._reasoning_mode_var.get() if self._reasoning_mode_var is not None else "auto"
        generation_backend = (
            self._generation_backend_var.get() if self._generation_backend_var is not None else "ollama"
        )
        embedding_backend = (
            self._embedding_backend_var.get()
            if self._embedding_backend_var is not None
            else "sentence_transformers"
        )
        vector_store_backend = self._vector_store_var.get() if self._vector_store_var is not None else "chromadb"
        web_provider = self._web_provider_var.get() if self._web_provider_var is not None else "wikipedia"
        allow_web_fallback = (
            bool(self._allow_web_fallback_var.get()) if self._allow_web_fallback_var is not None else True
        )
        enable_self_optimizer = (
            bool(self._enable_self_optimizer_var.get())
            if self._enable_self_optimizer_var is not None
            else False
        )
        reranking = bool(self._reranking_var.get()) if self._reranking_var is not None else True
        reranker_role_enabled = (
            bool(self._reranker_role_enabled_var.get())
            if self._reranker_role_enabled_var is not None
            else False
        )
        speech_to_text_role_enabled = (
            bool(self._speech_to_text_role_enabled_var.get())
            if self._speech_to_text_role_enabled_var is not None
            else False
        )
        text_to_speech_role_enabled = (
            bool(self._text_to_speech_role_enabled_var.get())
            if self._text_to_speech_role_enabled_var is not None
            else False
        )
        vad_role_enabled = bool(self._vad_role_enabled_var.get()) if self._vad_role_enabled_var is not None else False
        translation_role_enabled = (
            bool(self._translation_role_enabled_var.get())
            if self._translation_role_enabled_var is not None
            else False
        )
        code_specialist_role_enabled = (
            bool(self._code_specialist_role_enabled_var.get())
            if self._code_specialist_role_enabled_var is not None
            else False
        )
        long_horizon_enabled = (
            bool(self._long_horizon_enabled_var.get())
            if self._long_horizon_enabled_var is not None
            else False
        )
        long_horizon_minutes = (
            int((self._long_horizon_minutes_var.get() or "120").strip())
            if self._long_horizon_minutes_var is not None
            else 120
        )
        optimizer_policy = (
            self._optimizer_policy_var.get() if self._optimizer_policy_var is not None else "proposal_only"
        )
        optimizer_replay_limit = (
            int((self._optimizer_replay_limit_var.get() or "64").strip())
            if self._optimizer_replay_limit_var is not None
            else 64
        )
        show_debug_pane = (
            bool(self._show_debug_pane_var.get()) if self._show_debug_pane_var is not None else True
        )
        desktop_enabled = (
            bool(self._desktop_enabled_var.get()) if self._desktop_enabled_var is not None else False
        )
        desktop_approval_policy = (
            self._desktop_approval_policy_var.get()
            if self._desktop_approval_policy_var is not None
            else "approve_risky_only"
        )
        observation_tier = (
            self._observation_tier_var.get() if self._observation_tier_var is not None else "screenshot_on_demand"
        )
        cloud_mode = self._cloud_mode_var.get() if self._cloud_mode_var is not None else "auxiliary_only"
        log_runtime_events = (
            bool(self._log_runtime_events_var.get()) if self._log_runtime_events_var is not None else True
        )
        allow_cloud_content = (
            bool(self._allow_cloud_content_var.get()) if self._allow_cloud_content_var is not None else False
        )
        log_level = self._log_level_var.get() if self._log_level_var is not None else "INFO"
        enabled_roles = {
            str(item)
            for item in current.models.get("enabled_roles", ())
            if str(item).strip()
        }
        enabled_roles.update({"generation", "embedding"})
        if reranker_role_enabled:
            enabled_roles.add("reranker")
        else:
            enabled_roles.discard("reranker")
        if speech_to_text_role_enabled:
            enabled_roles.add("speech_to_text")
        else:
            enabled_roles.discard("speech_to_text")
        if text_to_speech_role_enabled:
            enabled_roles.add("text_to_speech")
        else:
            enabled_roles.discard("text_to_speech")
        if vad_role_enabled:
            enabled_roles.add("vad")
        else:
            enabled_roles.discard("vad")
        if translation_role_enabled:
            enabled_roles.add("translation")
        else:
            enabled_roles.discard("translation")
        if code_specialist_role_enabled:
            enabled_roles.add("code_specialist")
        else:
            enabled_roles.discard("code_specialist")
        profile = UserSettingsProfile(
            profile_name=profile_name,
            runtime={
                **current.runtime,
                "stub_mode": bool(current.runtime.get("stub_mode", True)),
                "allow_web_fallback": allow_web_fallback,
                "enable_self_optimizer": enable_self_optimizer,
                "generation_backend": generation_backend,
                "embedding_backend": embedding_backend,
                "vector_store_backend": vector_store_backend,
            },
            retrieval={
                **current.retrieval,
                "allow_web_fallback": allow_web_fallback,
                "provider": web_provider,
                "reranking": reranking,
            },
            reasoning={
                **current.reasoning,
                "thinking_minutes": thinking_minutes,
                "mode": reasoning_mode,
            },
            long_horizon={
                **current.long_horizon,
                "enabled": long_horizon_enabled,
                "wall_clock_minutes": long_horizon_minutes,
            },
            optimizer={
                **current.optimizer,
                "activation_policy": optimizer_policy,
                "replay_limit": optimizer_replay_limit,
            },
            models={
                **dict(current.models),
                "preferred_by_role": dict(current.models.get("preferred_by_role", {})),
                "enabled_roles": tuple(
                    [
                        role_name
                        for role_name in (
                            "generation",
                            "embedding",
                            "reranker",
                            "speech_to_text",
                            "vad",
                            "text_to_speech",
                            "translation",
                            "code_specialist",
                            "vision",
                            "specialist_perception",
                        )
                        if role_name in enabled_roles
                    ]
                    + sorted(
                        role_name
                        for role_name in enabled_roles
                        if role_name
                        not in {
                            "generation",
                            "embedding",
                            "reranker",
                            "speech_to_text",
                            "vad",
                            "text_to_speech",
                            "translation",
                            "code_specialist",
                            "vision",
                            "specialist_perception",
                        }
                    )
                ),
            },
            desktop={
                **current.desktop,
                "enabled": desktop_enabled,
                "approval_policy": desktop_approval_policy,
            },
            observation={
                **current.observation,
                "tier": observation_tier,
                "continuous_capture": observation_tier == "continuous_capture",
                "ocr_on_step": observation_tier in {"ocr_on_step", "vision_on_step", "continuous_capture"},
                "vision_on_step": observation_tier in {"vision_on_step", "continuous_capture"},
            },
            cloud={
                **current.cloud,
                "mode": cloud_mode,
                "enabled": cloud_mode != "disabled",
            },
            privacy={
                **current.privacy,
                "log_runtime_events": log_runtime_events,
                "allow_cloud_content": allow_cloud_content,
                "log_level": log_level,
            },
            ui={
                **current.ui,
                "show_debug_pane": show_debug_pane,
                "app_shell": "tkinter",
            },
        )
        profile.validate()
        return profile

    def _set_debug_pane_visibility(self, visible: bool) -> None:
        if self._debug_frame is None:
            return
        is_mapped = bool(self._debug_frame.winfo_manager())
        if visible and not is_mapped:
            self._debug_frame.pack(fill=tk.BOTH, expand=False, pady=(8, 0))
        elif not visible and is_mapped:
            self._debug_frame.pack_forget()

    def _on_thinking_minutes_changed(self, value: str) -> None:
        if self._thinking_label_var is None:
            return
        try:
            minutes = int(float(value))
        except (TypeError, ValueError):
            minutes = 0
        self._thinking_label_var.set(f"{minutes} minutes")
        self._refresh_time_control_summary()

    def _on_apply_time_preset(self, minutes: int) -> None:
        if self._thinking_minutes_var is not None:
            self._thinking_minutes_var.set(int(minutes))
        if self._thinking_label_var is not None:
            self._thinking_label_var.set(f"{int(minutes)} minutes")
        if self._long_horizon_enabled_var is not None:
            self._long_horizon_enabled_var.set(minutes > 120)
        if self._long_horizon_minutes_var is not None:
            self._long_horizon_minutes_var.set(str(int(minutes if minutes > 120 else 120)))
        if self._run_status_var is not None:
            mode_label = "long-horizon" if minutes > 120 else "interactive"
            self._run_status_var.set(f"Applied {mode_label} preset for {int(minutes)} minutes.")
        self._refresh_time_control_summary()

    def _on_pause_long_horizon_clicked(self) -> None:
        session_id = self._active_long_horizon_session_id()
        if not session_id:
            if self._run_status_var is not None:
                self._run_status_var.set("No active long-horizon session is available to pause.")
            return
        if self.request_action("long_horizon.pause", {"session_id": session_id}):
            if self._run_status_var is not None:
                self._run_status_var.set(f"Pause requested for long-horizon session '{session_id}'.")

    def _on_resume_long_horizon_clicked(self) -> None:
        session_id = self._active_long_horizon_session_id()
        if not session_id:
            if self._run_status_var is not None:
                self._run_status_var.set("No long-horizon session is selected for resume.")
            return
        if self.request_action("long_horizon.resume", {"session_id": session_id}):
            if self._run_status_var is not None:
                self._run_status_var.set(f"Resume requested for long-horizon session '{session_id}'.")

    def _on_cancel_long_horizon_clicked(self) -> None:
        session_id = self._active_long_horizon_session_id()
        if not session_id:
            if self._run_status_var is not None:
                self._run_status_var.set("No active long-horizon session is available to cancel.")
            return
        if self.request_action("long_horizon.cancel", {"session_id": session_id}):
            if self._run_status_var is not None:
                self._run_status_var.set(f"Cancel requested for long-horizon session '{session_id}'.")

    def _on_run_clicked(self) -> None:
        question = self._question_var.get().strip() if self._question_var is not None else ""
        thinking_minutes = int(self._thinking_minutes_var.get()) if self._thinking_minutes_var is not None else 30
        if not question:
            if self._run_status_var is not None:
                self._run_status_var.set("Enter a question before running a task.")
            return
        if self.request_task_submission(question, thinking_minutes):
            if self._run_status_var is not None:
                effective_minutes = self._planned_submission_minutes()
                if self._planned_execution_mode() == "long_horizon" and effective_minutes > thinking_minutes:
                    self._run_status_var.set(
                        f"Submitted long-horizon task with {effective_minutes} minute wall-clock budget."
                    )
                else:
                    self._run_status_var.set(f"Submitted task with {thinking_minutes} minute budget.")
        else:
            if self._run_status_var is not None:
                self._run_status_var.set("Run controller is unavailable in this session.")

    def _on_save_settings_clicked(self) -> None:
        try:
            profile = self._gather_settings_from_form()
        except (TypeError, ValueError) as exc:
            if self._settings_status_var is not None:
                self._settings_status_var.set(f"Settings validation failed: {exc}")
            return
        if self.request_settings_save(profile):
            if self._settings_status_var is not None:
                self._settings_status_var.set(f"Saved settings profile '{profile.profile_name}'.")
            self.request_action("settings.refresh_profiles", {"active_profile_name": profile.profile_name})
        else:
            if self._settings_status_var is not None:
                self._settings_status_var.set(
                    f"Applied settings profile '{profile.profile_name}' locally; persistence unavailable."
                )

    def _on_load_settings_clicked(self) -> None:
        profile_name = (self._profile_name_var.get().strip() if self._profile_name_var is not None else "") or "default"
        if self.request_action("settings.load_profile", {"profile_name": profile_name}):
            if self._settings_status_var is not None:
                self._settings_status_var.set(f"Loading saved settings profile '{profile_name}'.")
        elif self._settings_status_var is not None:
            self._settings_status_var.set("Profile loader is unavailable in this session.")

    def _on_refresh_settings_profiles_clicked(self) -> None:
        if self.request_action("settings.refresh_profiles", {"active_profile_name": self._profile_name_var.get() if self._profile_name_var is not None else "default"}):
            if self._settings_status_var is not None:
                self._settings_status_var.set("Refreshing saved settings profiles.")
        elif self._settings_status_var is not None:
            self._settings_status_var.set("Profile refresh is unavailable in this session.")

    def _on_import_settings_clicked(self) -> None:
        path = self._settings_path_var.get().strip() if self._settings_path_var is not None else ""
        if not path:
            if self._settings_status_var is not None:
                self._settings_status_var.set("Enter a JSON path before importing a settings profile.")
            return
        if self.request_action("settings.import_profile", {"path": path}):
            if self._settings_status_var is not None:
                self._settings_status_var.set(f"Importing settings profile from '{path}'.")
        elif self._settings_status_var is not None:
            self._settings_status_var.set("Profile import is unavailable in this session.")

    def _on_export_settings_clicked(self) -> None:
        path = self._settings_path_var.get().strip() if self._settings_path_var is not None else ""
        profile_name = (self._profile_name_var.get().strip() if self._profile_name_var is not None else "") or "default"
        if not path:
            if self._settings_status_var is not None:
                self._settings_status_var.set("Enter a JSON path before exporting a settings profile.")
            return
        if self.request_action("settings.export_profile", {"path": path, "profile_name": profile_name}):
            if self._settings_status_var is not None:
                self._settings_status_var.set(f"Exporting settings profile '{profile_name}' to '{path}'.")
        elif self._settings_status_var is not None:
            self._settings_status_var.set("Profile export is unavailable in this session.")

    def _on_reset_settings_clicked(self) -> None:
        self._apply_settings_to_form(self._app_state.user_settings)
        if self._settings_status_var is not None:
            self._settings_status_var.set("Reset form to the persisted profile state.")

    def _on_refresh_history_clicked(self) -> None:
        if self.request_action("history.refresh", {}):
            if self._app_notice_var is not None:
                self._app_notice_var.set("Refreshing task history.")

    def _on_refresh_examples_clicked(self) -> None:
        sample_id = self._sample_task_id_var.get().strip() if self._sample_task_id_var is not None else ""
        if self.request_action("examples.refresh", {"sample_id": sample_id}):
            if self._examples_status_var is not None:
                self._examples_status_var.set("Refreshing Phase 11 examples.")

    def _on_load_demo_pack_clicked(self) -> None:
        if self.request_action("examples.load_demo_pack", {}):
            if self._examples_status_var is not None:
                self._examples_status_var.set("Loading the Phase 11 demo pack into local storage.")

    def _on_load_sample_clicked(self) -> None:
        sample_id = self._sample_task_id_var.get().strip() if self._sample_task_id_var is not None else ""
        if not sample_id:
            if self._examples_status_var is not None:
                self._examples_status_var.set("Enter or select a sample ID first.")
            return
        if self.request_action("examples.select_sample", {"sample_id": sample_id}):
            if self._examples_status_var is not None:
                self._examples_status_var.set(f"Loading sample '{sample_id}'.")

    def _on_run_sample_clicked(self) -> None:
        sample_id = self._sample_task_id_var.get().strip() if self._sample_task_id_var is not None else ""
        if not sample_id:
            if self._examples_status_var is not None:
                self._examples_status_var.set("Enter or select a sample ID before running it.")
            return
        if self.request_action("examples.run_sample_task", {"sample_id": sample_id}):
            if self._examples_status_var is not None:
                self._examples_status_var.set(f"Running sample '{sample_id}'.")

    def _on_export_verified_trace_clicked(self) -> None:
        path = self._sample_export_path_var.get().strip() if self._sample_export_path_var is not None else ""
        if not path:
            if self._examples_status_var is not None:
                self._examples_status_var.set("Enter an export path before exporting the verified trace example.")
            return
        if self.request_action("examples.export_verified_trace_example", {"path": path}):
            if self._examples_status_var is not None:
                self._examples_status_var.set(f"Exporting the packaged verified trace example to '{path}'.")

    def _on_inspect_task_clicked(self) -> None:
        task_id = self._history_task_id_var.get().strip() if self._history_task_id_var is not None else ""
        if not task_id:
            if self._app_notice_var is not None:
                self._app_notice_var.set("Enter a task ID before loading the run inspector.")
            return
        if self.request_action("history.inspect_task", {"task_id": task_id}):
            if self._app_notice_var is not None:
                self._app_notice_var.set(f"Loading task '{task_id}'.")

    def _on_export_task_debug_clicked(self) -> None:
        task_id = self._history_task_id_var.get().strip() if self._history_task_id_var is not None else ""
        path = self._history_export_path_var.get().strip() if self._history_export_path_var is not None else ""
        if not task_id or not path:
            if self._app_notice_var is not None:
                self._app_notice_var.set("Enter both a task ID and export path before exporting trace debug.")
            return
        if self.request_action("history.export_task_debug", {"task_id": task_id, "path": path}):
            if self._app_notice_var is not None:
                self._app_notice_var.set(f"Exporting trace debug for '{task_id}'.")

    def _on_refresh_knowledge_clicked(self) -> None:
        if self.request_action("knowledge.refresh", {}):
            if self._knowledge_status_var is not None:
                self._knowledge_status_var.set("Refreshing knowledge library.")

    def _on_ingest_knowledge_clicked(self) -> None:
        source_ref = self._knowledge_source_ref_var.get().strip() if self._knowledge_source_ref_var is not None else ""
        title = self._knowledge_title_var.get().strip() if self._knowledge_title_var is not None else ""
        content = self._knowledge_content_text.get("1.0", "end").strip() if self._knowledge_content_text is not None else ""
        if not source_ref or not content:
            if self._knowledge_status_var is not None:
                self._knowledge_status_var.set("Enter a source ref and document text before ingesting.")
            return
        if self.request_action(
            "knowledge.ingest_text",
            {"source_ref": source_ref, "title": title, "content": content},
        ):
            if self._knowledge_status_var is not None:
                self._knowledge_status_var.set(f"Ingesting '{source_ref}'.")

    def _on_archive_knowledge_clicked(self) -> None:
        self._request_knowledge_action("knowledge.archive_source", "Archiving source.")

    def _on_unarchive_knowledge_clicked(self) -> None:
        self._request_knowledge_action("knowledge.unarchive_source", "Restoring source.")

    def _on_rebuild_knowledge_clicked(self) -> None:
        self._request_knowledge_action("knowledge.rebuild_source", "Rebuilding source embeddings.")

    def _on_remove_knowledge_clicked(self) -> None:
        self._request_knowledge_action("knowledge.remove_source", "Removing source.")

    def _request_knowledge_action(self, action: str, status_text: str) -> None:
        source_ref = self._knowledge_source_ref_var.get().strip() if self._knowledge_source_ref_var is not None else ""
        if not source_ref:
            if self._knowledge_status_var is not None:
                self._knowledge_status_var.set("Enter a source ref first.")
            return
        if self.request_action(action, {"source_ref": source_ref}):
            if self._knowledge_status_var is not None:
                self._knowledge_status_var.set(status_text)

    def _selected_model_role_value(self) -> str:
        if self._model_role_var is None:
            return ModelRole.GENERATION.value
        return str(self._model_role_var.get()).strip() or ModelRole.GENERATION.value

    def _request_model_role_action(self, action: str, status_text: str) -> None:
        role = self._selected_model_role_value()
        if self.request_action(action, {"role": role}):
            if self._model_role_status_var is not None:
                self._model_role_status_var.set(status_text)
        elif self._model_role_status_var is not None:
            self._model_role_status_var.set("Local-AI control-plane actions are unavailable in this session.")

    def _on_model_install_guidance_clicked(self) -> None:
        self._request_model_role_action(
            "model.install_guidance",
            f"Loading install guidance for '{self._selected_model_role_value()}'.",
        )

    def _on_enable_model_role_clicked(self) -> None:
        self._request_model_role_action(
            "model.enable_role",
            f"Enabling '{self._selected_model_role_value()}'.",
        )

    def _on_disable_model_role_clicked(self) -> None:
        self._request_model_role_action(
            "model.disable_role",
            f"Disabling '{self._selected_model_role_value()}'.",
        )

    def _on_warm_model_role_clicked(self) -> None:
        self._request_model_role_action(
            "model.warm_role",
            f"Warming '{self._selected_model_role_value()}'.",
        )

    def _on_unload_model_role_clicked(self) -> None:
        self._request_model_role_action(
            "model.unload_role",
            f"Unloading '{self._selected_model_role_value()}'.",
        )

    def _on_test_model_role_clicked(self) -> None:
        self._request_model_role_action(
            "model.test_ping",
            f"Testing '{self._selected_model_role_value()}'.",
        )

    def _on_inspect_model_fallback_clicked(self) -> None:
        self._request_model_role_action(
            "model.inspect_fallback",
            f"Inspecting fallback state for '{self._selected_model_role_value()}'.",
        )

    def _on_transcribe_audio_clicked(self) -> None:
        path = self._audio_path_var.get().strip() if self._audio_path_var is not None else ""
        if not path:
            if self._audio_status_var is not None:
                self._audio_status_var.set("Enter a local .wav path before transcribing audio.")
            return
        if self.request_action("audio.transcribe_file", {"path": path}):
            if self._audio_status_var is not None:
                self._audio_status_var.set(f"Transcribing '{path}'.")

    def _on_use_audio_transcript_clicked(self) -> None:
        if self.request_action("audio.use_transcript_as_question", {}):
            if self._audio_status_var is not None:
                self._audio_status_var.set("Applying the current transcript to the question box.")

    def _on_synthesize_audio_text_clicked(self) -> None:
        text = self._tts_text_var.get().strip() if self._tts_text_var is not None else ""
        output_path = self._tts_output_path_var.get().strip() if self._tts_output_path_var is not None else ""
        if not text or not output_path:
            if self._audio_status_var is not None:
                self._audio_status_var.set("Enter text and a local .wav output path before synthesizing speech.")
            return
        if self.request_action("audio.synthesize_text", {"text": text, "path": output_path}):
            if self._audio_status_var is not None:
                self._audio_status_var.set(f"Synthesizing voice output to '{output_path}'.")

    def _on_speak_answer_clicked(self) -> None:
        output_path = self._tts_output_path_var.get().strip() if self._tts_output_path_var is not None else ""
        if not output_path:
            if self._audio_status_var is not None:
                self._audio_status_var.set("Enter a local .wav output path before speaking the current answer.")
            return
        if self.request_action("audio.speak_answer", {"path": output_path}):
            if self._audio_status_var is not None:
                self._audio_status_var.set(f"Synthesizing the current answer to '{output_path}'.")

    def _on_clear_audio_clicked(self) -> None:
        if self.request_action("audio.clear", {}):
            if self._audio_status_var is not None:
                self._audio_status_var.set("Clearing the current audio transcript.")

    def _on_clear_audio_output_clicked(self) -> None:
        if self.request_action("audio.clear_output", {}):
            if self._audio_status_var is not None:
                self._audio_status_var.set("Clearing the current voice output state.")

    def _on_translate_text_clicked(self) -> None:
        text = self._translation_input_var.get().strip() if self._translation_input_var is not None else ""
        source_language = (
            self._translation_source_language_var.get().strip()
            if self._translation_source_language_var is not None
            else self.config.translation.default_source_language
        )
        target_language = (
            self._translation_target_language_var.get().strip()
            if self._translation_target_language_var is not None
            else self.config.translation.default_target_language
        )
        if not text:
            if self._translation_status_var is not None:
                self._translation_status_var.set("Enter text before translating.")
            return
        if self.request_action(
            "translation.translate_text",
            {
                "text": text,
                "source_language": source_language,
                "target_language": target_language,
            },
        ):
            if self._translation_status_var is not None:
                self._translation_status_var.set(
                    f"Translating text from {source_language or '(auto)'} to {target_language or '(none)'}."
                )

    def _on_translate_answer_clicked(self) -> None:
        source_language = (
            self._translation_source_language_var.get().strip()
            if self._translation_source_language_var is not None
            else self.config.translation.default_source_language
        )
        target_language = (
            self._translation_target_language_var.get().strip()
            if self._translation_target_language_var is not None
            else self.config.translation.default_target_language
        )
        if self.request_action(
            "translation.translate_answer",
            {
                "source_language": source_language,
                "target_language": target_language,
            },
        ):
            if self._translation_status_var is not None:
                self._translation_status_var.set(
                    f"Translating the current answer to {target_language or '(none)'}."
                )

    def _on_use_translation_clicked(self) -> None:
        if self.request_action("translation.use_as_question", {}):
            if self._translation_status_var is not None:
                self._translation_status_var.set("Applying the current translation to the question box.")

    def _on_clear_translation_clicked(self) -> None:
        if self.request_action("translation.clear", {}):
            if self._translation_status_var is not None:
                self._translation_status_var.set("Clearing the current translation result.")

    def _on_analyze_code_file_clicked(self) -> None:
        source_path = self._code_source_path_var.get().strip() if self._code_source_path_var is not None else ""
        request_text = (
            self._code_request_var.get().strip() if self._code_request_var is not None else self.config.code_specialist.default_request
        )
        if not source_path:
            if self._code_status_var is not None:
                self._code_status_var.set("Enter a local file path before analyzing code.")
            return
        if self.request_action(
            "code.analyze_file",
            {"path": source_path, "request_text": request_text},
        ):
            if self._code_status_var is not None:
                self._code_status_var.set(f"Analyzing '{source_path}'.")

    def _on_analyze_code_snippet_clicked(self) -> None:
        snippet = self._code_input_text.get("1.0", "end").strip() if self._code_input_text is not None else ""
        request_text = (
            self._code_request_var.get().strip() if self._code_request_var is not None else self.config.code_specialist.default_request
        )
        if not snippet:
            if self._code_status_var is not None:
                self._code_status_var.set("Paste a code snippet before analyzing it.")
            return
        if self.request_action(
            "code.analyze_snippet",
            {"text": snippet, "request_text": request_text},
        ):
            if self._code_status_var is not None:
                self._code_status_var.set("Analyzing the current snippet.")

    def _on_clear_code_clicked(self) -> None:
        if self.request_action("code.clear", {}):
            if self._code_status_var is not None:
                self._code_status_var.set("Clearing the current code-specialist result.")

    def _on_refresh_readiness_clicked(self) -> None:
        if self.request_action("readiness.refresh", {}):
            if self._app_notice_var is not None:
                self._app_notice_var.set("Refreshing readiness report.")

    def _on_window_close(self) -> None:
        self._stop_flag.set()
        if self._root is not None:
            self._root.quit()

    @staticmethod
    def _format_gb(value: float | None) -> str:
        if value is None:
            return "n/a"
        return f"{value:.2f}GB"
