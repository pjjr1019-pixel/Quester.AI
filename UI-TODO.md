# Next-Generation Local AI Shell for Windows Desktop

## Goal
- Transform the app into a premium, orb-centered PySide6 shell that feels like a living local AI presence.
- Preserve all backend logic, typed contracts, orchestration, storage, settings, history, routing, long-horizon reasoning, capability control, and packaged Windows behavior.
- Treat this as a shell migration and integration project, not a backend rewrite.

## Hard Rules
- [ ] Do not rewrite core backend flow unless a small additive adapter layer is required.
- [ ] Keep `Orchestrator.run_task(question, thinking_minutes)` as the main execution contract.
- [ ] Keep `TaskResult`, `RuntimeEvent`, `AgentStatus`, `ResourceBudget`, and existing dataclass compatibility intact.
- [ ] Keep headless mode valid while the new shell is built.
- [ ] Keep packaged Windows startup, stub mode, readiness/preflight, and recovery flows intact.
- [ ] Keep all advanced current features reachable, even if they move into drawers or sheets.
- [ ] Keep low-resource and reduced-effects modes intentional rather than visually broken.

## Preserve
- [ ] Planner, Researcher, Reasoner, Critic, Compressor, SelfOptimizer
- [ ] Long-horizon scheduling, checkpointing, pause, resume, cancel
- [ ] Task history, run inspector, knowledge library, settings, profiles
- [ ] Readiness/preflight, onboarding, packaged startup diagnostics
- [ ] Local-AI control plane, model registry, routing, fallback visibility
- [ ] Capability registry, policy engine, local task sessions, approval flow
- [ ] Observation tiers, OCR, vision, optional cloud helpers
- [ ] Stub mode, real mode, existing tests, compatibility expectations

## Current Status
- [x] Added typed shell-facing state models and projection foundations.
- [x] Added `shell_state_snapshot()` to the dashboard service layer.
- [x] Added optional PySide6 dependency hooks and a real optional PySide6 shell path.
- [x] Added initial shell-state tests and PySide6 availability tests.
- [x] Real PySide6 shell can now launch when `app_shell="pyside6"` and PySide6 is installed.
- [ ] Advanced dashboard surfaces are not yet migrated into PySide6 drawers and sheets.
- [ ] Orb rendering is still foundation-only and not feature-complete.

## Final UX Definition
- [ ] Large animated orb in the top hero region as the main focal point.
- [ ] Clear state-driven status and sub-status directly below the orb.
- [ ] Bounded, explainable activity strip showing what the system is doing now.
- [ ] Premium conversation/task surface centered around active work and verified answers.
- [ ] Bottom input dock with message entry, send, mode controls, and task controls.
- [ ] Left operator drawer for agent monitor, session controls, and timeline.
- [ ] Right operator drawer for evidence, provenance, compressor, optimizer, runtime health, and local-AI control plane.
- [ ] Lower sheets for history, knowledge, settings, readiness, capabilities, and debug.
- [ ] Whole-shell reactivity: background tint, panel accents, resource ribbon, notifications, and state-aware polish.

## Phase 1 - Backend/UI Boundary
- [x] Confirm the backend remains event-driven and task-driven.
- [x] Keep `DashboardService` as the UI-facing bridge instead of moving logic into widgets.
- [x] Refactor `DashboardService` so it acts as a shell-agnostic state/controller facade rather than a Tkinter-specific host.
- [x] Split renderer lifecycle from app-state lifecycle.
- [ ] Define one clean UI host interface for:
  - [ ] start shell
  - [ ] stop shell
  - [ ] refresh from latest app state
  - [ ] submit task
  - [ ] request action
  - [ ] open sheet or drawer
- [ ] Remove direct Tkinter assumptions from shared dashboard state handling.
- [ ] Keep old settings payloads loadable without migration breakage.

## Phase 2 - Typed ShellState Projection
- [x] Add typed shell projection models.
- [ ] Audit the shell projection model and fill any missing fields needed by the new shell:
  - [ ] `orb_mode`
  - [ ] `orb_palette`
  - [ ] `orb_intensity`
  - [ ] `ring_mode`
  - [ ] `particle_mode`
  - [ ] `ambient_mode`
  - [ ] `status_text`
  - [ ] `sub_status_text`
  - [ ] `active_agent`
  - [ ] `secondary_agents`
  - [ ] `active_tools`
  - [ ] `active_roles`
  - [ ] `confidence_band`
  - [ ] `verifier_state`
  - [ ] `retrieval_state`
  - [ ] `compression_state`
  - [ ] `optimizer_state`
  - [ ] `long_horizon_state`
  - [ ] `checkpoint_count`
  - [ ] `degraded_reason`
  - [ ] `fallback_reason`
  - [ ] `resource_pressure_level`
  - [ ] `speaking_state`
  - [ ] `approval_pending`
  - [ ] `capability_session_state`
  - [ ] `observation_tier`
  - [ ] `cloud_helper_state`
  - [ ] transient effect flags
  - [ ] shell notifications
  - [ ] current task summary
- [ ] Lock shell-state priority rules:
  - [ ] error overrides all primary modes
  - [ ] approval pending overlays without erasing base mode
  - [ ] deep reasoning overrides normal thinking
  - [ ] degraded overlays without erasing the primary state
  - [ ] transient effects are bounded and expire automatically
- [ ] Map shell state from:
  - [ ] `RuntimeEvent`
  - [ ] `AgentStatus`
  - [ ] runtime health snapshots
  - [ ] model route decisions
  - [ ] `TaskResult`
  - [ ] checkpoint events
  - [ ] capability/session events
  - [ ] readiness/preflight state
- [x] Make shell widgets consume only `ShellState`, not raw event dicts.

## Phase 3 - PySide6 App Host
- [x] Add optional PySide6 dependency declaration.
- [x] Install and validate PySide6 in the source environment.
- [x] Add a real Qt app bootstrap path.
- [ ] Add a shell host entrypoint for source runs.
- [ ] Add a shell host entrypoint for packaged runs.
- [ ] Ensure packaged launch still starts in stub mode when real-mode prerequisites are unavailable.
- [x] Keep clean failure messaging if PySide6 is requested but not installed.
- [x] Keep headless mode usable without importing UI runtime modules.
- [x] Ensure Qt event loop and orchestrator lifecycle cooperate cleanly.
- [x] Ensure stop/shutdown tears down timers, background polling, and shell resources cleanly.

## Phase 4 - Visual Design System
- [ ] Define a shell theme token layer:
  - [ ] background gradients
  - [ ] orb palettes by state
  - [ ] panel fills
  - [ ] glass overlay colors
  - [ ] accent rails
  - [ ] text hierarchy
  - [ ] warning/success/error colors
  - [ ] chip colors
  - [ ] timeline marker colors
- [ ] Define PySide6 stylesheet tokens for shared components.
- [ ] Choose Windows-safe local fonts and fallbacks.
- [ ] Define spacing, corner radius, border glow, and shadow rules.
- [ ] Define animation timing tokens:
  - [ ] idle
  - [ ] active
  - [ ] speaking
  - [ ] long-horizon
  - [ ] reduced motion
- [ ] Define presets:
  - [ ] Minimal
  - [ ] Balanced
  - [ ] Immersive

## Phase 5 - OrbWidget Core
- [x] Build `OrbWidget` as a standalone custom-painted component.
- [ ] Add layered rendering:
  - [ ] core
  - [ ] inner shimmer
  - [ ] shell surface
  - [ ] outer glow
  - [ ] halo ring
  - [ ] reflection bed
- [ ] Add base animation engine:
  - [ ] breathing
  - [ ] floating
  - [ ] pulse interpolation
  - [ ] palette transitions
  - [ ] intensity smoothing
- [ ] Add demo/test states to exercise every orb mode without backend activity.
- [ ] Tune performance for desktop window resizing and idle rendering.

## Phase 6 - Advanced Orb Effects
- [x] Add segmented tool ring.
- [ ] Add role constellation markers.
- [x] Add confidence arc.
- [ ] Add speaking waveform halo.
- [ ] Add critic verification sweep.
- [ ] Add compressor contraction pulse.
- [x] Add checkpoint pulse for long-horizon saves.
- [x] Add approval-pending hold ring.
- [ ] Add optimizer advisory aura.
- [ ] Add degraded undertone overlay.
- [ ] Add transient effects:
  - [ ] insight flash
  - [ ] consensus shimmer
  - [ ] verification lock cue
- [ ] Add bounded particle modes:
  - [ ] idle sparse particles
  - [ ] retrieval inward particles
  - [ ] deep-mode dense orbit particles
  - [ ] speaking reactive halo particles
- [ ] Add an effect priority system so the orb stays readable under multiple simultaneous signals.
- [ ] Add reduced-effects versions of every effect that matters.

## Phase 7 - Main Shell Window
- [x] Build the main `QMainWindow` shell.
- [x] Create layout regions:
  - [x] top hero region
  - [x] center task/conversation region
  - [x] bottom control dock
  - [x] left drawer
  - [x] right drawer
  - [x] lower sheets/modals
- [ ] Add resize-aware layout behavior for wide desktop, mid-width, and compact window sizes.
- [ ] Keep the orb dominant on all supported window sizes.
- [ ] Ensure drawers do not visually overpower the hero/task flow.

## Phase 8 - Hero Region
- [x] Place the orb in the upper third of the window.
- [x] Add primary status label below the orb.
- [x] Add sub-status label below the primary status.
- [ ] Add readable cognitive posture wording:
  - [ ] Ready
  - [ ] Exploring
  - [ ] Focusing
  - [ ] Verifying
  - [ ] Refining
  - [ ] Advising
  - [ ] Waiting
- [ ] Map status and sub-status text from real shell state.
- [ ] Add state-aware transitions between status changes.
- [ ] Add subtle background haze and light reflection tied to orb color.

## Phase 9 - Activity Strip
- [x] Build a bounded activity-chip strip below the hero status.
- [ ] Support subsystem chips such as:
  - [ ] Local Retrieval
  - [ ] Web Query
  - [ ] Verifying
  - [ ] Deep Candidate Pass
  - [ ] Compression Review
  - [ ] Optimizer Advisory
  - [ ] Fallback Active
  - [ ] Approval Needed
  - [ ] Capability Session
  - [ ] Observation Tier Active
  - [ ] Cloud Helper Ready
  - [ ] Checkpoint Saved
  - [ ] Resource Pressure
  - [ ] Route Changed
- [ ] Add subtle active animations to live chips.
- [x] Add hover tooltips for detail text.
- [ ] Add chip overflow handling with a compact "more" pattern.
- [ ] Add reduced-motion behavior.

## Phase 10 - Conversation and Task Surface
- [x] Build a modern conversation/task stream.
- [x] Create a user message card style.
- [x] Create an assistant answer card style.
- [x] Create system/tool/status card styles.
- [x] Create an active task card that shows:
  - [x] current phase
  - [x] elapsed time
  - [x] candidate count
  - [x] evidence count
  - [x] verification state
  - [x] fallback status
  - [x] confidence band
  - [x] route summary
- [x] Create a final answer card that shows:
  - [x] final answer
  - [x] evidence summary
  - [x] citations
  - [x] warnings
  - [x] verification result
  - [x] degraded or uncertainty notes
  - [x] export shortcuts
- [x] Add expandable sections:
  - [x] Why this answer
  - [x] How it was verified
  - [x] What deep mode changed
- [ ] Add entry transitions for new cards.
- [ ] Keep auto-scroll bounded and non-jarring.
- [ ] Keep the surface readable without overpowering the orb.

## Phase 11 - Input Dock and Task Controls
- [x] Build a premium bottom input dock.
- [ ] Add:
  - [x] mic/voice button
  - [x] text input
  - [x] send button
  - [x] thinking-time slider
  - [x] quick mode buttons
  - [x] stop button during active runs
  - [x] pause/resume for long-horizon runs
- [ ] Add quick mode buttons:
  - [x] Fast
  - [x] Deep
  - [x] Long Horizon
- [ ] Add optional compact toggles:
  - [x] Web Allowed
  - [x] Verification Priority
  - [x] Local Only
  - [x] Capability Session Enabled
  - [x] Cloud Helper Allowed
- [ ] Add state-reactive glow and focus styling.
- [ ] Add keyboard shortcuts.
- [ ] Keep disabled and busy states readable.
- [ ] Keep voice affordances visible even when STT/TTS roles are unavailable.

## Phase 12 - Left Drawer
- [x] Build a left-side operator drawer.
- [x] Move agent monitor into the drawer.
- [x] Add session controls.
- [x] Add task timeline tray.
- [x] Add compact diagnostics for the current task.
- [x] Add pause/resume/cancel shortcuts.
- [ ] Add open/close transitions that do not stall the main shell.

## Phase 13 - Right Drawer
- [x] Build a right-side operator drawer.
- [x] Move evidence inspector into the drawer.
- [x] Move macro/provenance inspector into the drawer.
- [x] Move compressor insights into the drawer.
- [x] Move optimizer suggestions into the drawer.
- [x] Move runtime health into the drawer.
- [x] Move local-AI control plane into the drawer.
- [ ] Keep each section readable in compact and expanded states.

## Phase 14 - Timeline Tray
- [ ] Build a timeline tray for the current task.
- [ ] Show milestone entries such as:
  - [ ] Planner started
  - [ ] Local retrieval complete
  - [ ] Web fallback triggered
  - [ ] Candidate batch `n/m`
  - [ ] Critic verification pass
  - [ ] Compressor review complete
  - [ ] Answer verified
  - [ ] Checkpoint saved
  - [ ] Optimizer advisory considered
  - [ ] Final answer rendered
- [ ] Show retries, fallbacks, degraded branches, pauses, and resumptions.
- [ ] Keep the tray collapsible and persistent for the current session.
- [ ] Make timeline snapshots exportable into task history.

## Phase 15 - Long-Horizon Tray
- [x] Build a dedicated long-horizon session tray.
- [ ] Show:
  - [x] elapsed time
  - [x] current cycle type
  - [x] checkpoint count
  - [x] candidate growth
  - [x] verification passes
  - [x] evidence refresh count
  - [x] confidence improvement trend
  - [x] advisory suggestions considered
  - [x] early-stop reason
- [x] Add pause/resume/cancel controls.
- [ ] Add a duty-cycle visualization.
- [ ] Add resource-throttle notices when long-horizon work is intentionally slowed.
- [x] Add "what extra time bought" summaries.

## Phase 16 - Local-AI Control Plane
- [ ] Turn the existing model and routing surface into a flagship panel.
- [ ] Show:
  - [x] installed local roles
  - [x] active routed roles
  - [x] current routing decision
  - [x] heavy-slot usage
  - [ ] sidecar helpers
  - [x] fallback reasons
  - [x] readiness gaps
  - [x] route history
  - [ ] optimizer suggestions relevant to routing
- [ ] Add quick actions:
  - [ ] warm model
  - [ ] unload model
  - [ ] enable role
  - [ ] disable role
  - [ ] test ping
  - [ ] inspect fallback reason
  - [ ] open readiness guidance
- [x] Add structured route summaries to the active task card and answer card where relevant.

## Phase 17 - Capability and Session UX
- [x] Surface compact capability context in the main shell:
  - [x] active session badge
  - [x] target app/window badge
  - [x] pending approval pill
  - [x] observation tier badge
  - [x] last action preview
  - [x] policy state summary
- [ ] Keep full capability registry and audit trail in secondary views.
- [x] Build shell-consistent approval prompts.
- [x] Make approval prompts clearly explain:
  - [x] what action is requested
  - [x] why approval is needed
  - [x] what the target is
  - [x] what the risk level is
- [x] Keep approval state visible in the orb and activity strip.

## Phase 18 - Secondary Sheets and Views
- [x] Build lower sheets or modal views for:
  - [x] task history
  - [x] run inspector
  - [x] knowledge library
  - [x] settings
  - [x] readiness/preflight
  - [x] capability details
  - [x] debug/event log
- [ ] Restyle existing surfaces to match the new shell without removing functionality.
- [x] Keep export and import actions reachable.
- [ ] Keep debug raw-event access available but de-emphasized.

## Phase 19 - Shell-Wide Reactivity
- [ ] Add state-aware background tinting.
- [ ] Add subtle animated haze behind the orb.
- [ ] Add panel accent rails tied to shell state.
- [ ] Add state-aware button hover and focus styling.
- [ ] Add answer-card verification accents.
- [ ] Add fallback and degraded warning styling.
- [ ] Add shell notification animations.
- [x] Add a compact integrated resource ribbon that shows:
  - [x] RAM
  - [x] VRAM
  - [x] heavy slots used
  - [ ] degraded mode flag
  - [x] optional cloud-helper flag
  - [x] observation tier flag
- [ ] Keep shell motion meaningful rather than decorative.

## Phase 20 - Settings, Presets, and Accessibility
- [ ] Add persisted visual settings:
  - [x] orb size
  - [x] animation intensity
  - [x] reduced motion
  - [x] ambient reactivity
  - [x] particle density
  - [x] side drawer defaults
  - [x] activity strip visibility
  - [x] task timeline visibility
  - [x] resource ribbon visibility
  - [x] notification visibility
- [ ] Add persisted performance settings:
  - [x] low-resource mode
  - [x] reduced-effects mode
  - [x] animation frame cap
  - [x] simplified orb mode
- [ ] Add presets:
  - [x] Minimal
  - [x] Balanced
  - [x] Immersive
- [ ] Add accessibility controls:
  - [x] low motion
  - [x] higher contrast
  - [x] larger status text
  - [x] simpler accent behavior
- [x] Keep settings import/export compatible with existing profiles.

## Phase 21 - Performance and Rendering Bounds
- [ ] Add a shared animation clock strategy rather than per-widget free-running loops.
- [ ] Define frame caps for:
  - [ ] standard mode
  - [ ] reduced-effects mode
  - [ ] low-resource mode
- [ ] Cap particle counts by preset.
- [ ] Reduce effect complexity automatically when low-resource mode is enabled.
- [ ] Keep runtime pressure visible but do not silently change user visual presets.
- [ ] Profile expensive painting paths in the orb and background.
- [ ] Ensure resize, drawer animation, and event bursts remain responsive.

## Phase 22 - Packaging and Startup
- [x] Update packaged app entrypoints so the PySide6 shell becomes the default UI.
- [ ] Keep source-from-repo workflow valid.
- [x] Ensure first-run onboarding matches the new shell language and layout.
- [x] Ensure preflight/readiness can be opened from the new shell.
- [ ] Ensure startup failures still degrade safely to stub mode with readable guidance.
- [x] Ensure diagnostics and support bundle export remain reachable.
- [ ] Ensure packaged build includes the required PySide6 runtime pieces.

## Phase 23 - Regression and Test Coverage
- [x] Keep all current backend tests green.
- [x] Add shell-state mapping tests for:
  - [x] idle
  - [x] listening
  - [x] planning
  - [x] local retrieval
  - [x] web retrieval
  - [x] fast reasoning
  - [x] deep reasoning
  - [x] critic verification
  - [x] compression
  - [x] responding
  - [x] speaking
  - [x] degraded mode
  - [x] approval pending
  - [x] error
  - [x] offline
- [x] Add PySide6 widget tests for:
  - [x] shell launch
  - [x] shell shutdown
  - [x] orb demo states
  - [x] drawer open/close
  - [x] timeline population
  - [x] conversation/task cards
  - [x] resource ribbon
  - [x] settings persistence
  - [x] reduced-effects mode
  - [x] low-resource mode
  - [x] resize behavior
- [ ] Add flow tests for:
  - [x] stub-mode launch
  - [x] real-mode launch
  - [x] fast task
  - [x] deep task
  - [x] long-horizon task
  - [x] degraded/fallback path
  - [x] web fallback
  - [x] critic rejection
  - [x] compressor activity
  - [x] optimizer advisory visibility
  - [x] task history access
  - [x] knowledge library access
  - [x] settings/profile round-trip
  - [x] readiness/preflight flow
  - [x] local-AI control plane flow
  - [x] capability session visibility
  - [x] approval flow
  - [x] packaged Windows launch
  - [x] event burst throttling
- [x] Run Qt tests in offscreen mode for CI stability.

## Phase 24 - Rollout Order
- [ ] Step 1: finish the backend/UI host split.
- [ ] Step 2: complete the `ShellState` projection contract.
- [ ] Step 3: finish a fully state-driven standalone `OrbWidget`.
- [ ] Step 4: build the new PySide6 main shell frame.
- [ ] Step 5: add status, sub-status, and activity strip.
- [ ] Step 6: build the conversation/task surface and input dock.
- [ ] Step 7: migrate advanced dashboard features into drawers and sheets.
- [ ] Step 8: wire live runtime events and state updates into the shell.
- [ ] Step 9: add advanced orb effects and the effect-priority system.
- [ ] Step 10: add long-horizon and timeline trays.
- [ ] Step 11: upgrade the local-AI control plane into a premium operator surface.
- [ ] Step 12: integrate capability/session visuals and approval overlays.
- [ ] Step 13: add shell-wide background, panel, and resource reactivity.
- [ ] Step 14: add settings, presets, accessibility, and performance controls.
- [ ] Step 15: run regression and packaged-app validation.
- [ ] Step 16: polish motion, copy, spacing, responsiveness, and performance.

## Immediate Next Work
- [ ] Finish the explicit UI-host interface so sheet and drawer navigation can be commanded cleanly from the service layer.
- [ ] Replace the read-only right-drawer summaries with fully interactive evidence, provenance, compressor, optimizer, and control-plane widgets.
- [ ] Build dedicated active-task and final-answer cards instead of using the generic conversation card layout for every role.
- [ ] Add the missing long-horizon tray, richer timeline details, and capability-session badges in the main shell surface.
- [ ] Expand PySide widget coverage to include drawer navigation, resize behavior, reduced-effects mode, and more shell-state transitions without relying on cross-file Qt singleton reuse.

## Blockers to Watch
- [x] PySide6 is now installed in the working source environment for live shell verification.
- [ ] Qt event loop integration must not break headless mode or packaged startup behavior.
- [ ] Shell migration must not regress existing readiness, history, knowledge, settings, or task controls.
- [ ] Rich visual rendering must remain bounded on the target 6GB VRAM / 8GB RAM profile.

## Done Criteria
- [ ] The app still preserves all current backend features and workflows.
- [ ] The main user experience is orb-centered and clearly state-driven.
- [ ] The orb reflects real runtime behavior rather than fake decorative motion.
- [ ] Advanced system depth is accessible through clean drawers and sheets.
- [ ] Long-horizon reasoning is understandable and visibly valuable.
- [ ] The local-AI control plane feels like a first-class feature.
- [ ] Capability sessions and approvals are visible without cluttering the shell.
- [ ] Reduced-effects and low-resource modes still look premium.
- [ ] The packaged Windows app feels like a real local AI shell rather than a developer dashboard.
