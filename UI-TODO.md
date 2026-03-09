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
- [ ] Create an active task card that shows:
  - [ ] current phase
  - [ ] elapsed time
  - [ ] candidate count
  - [ ] evidence count
  - [ ] verification state
  - [ ] fallback status
  - [ ] confidence band
  - [ ] route summary
- [ ] Create a final answer card that shows:
  - [ ] final answer
  - [ ] evidence summary
  - [ ] citations
  - [ ] warnings
  - [ ] verification result
  - [ ] degraded or uncertainty notes
  - [ ] export shortcuts
- [ ] Add expandable sections:
  - [ ] Why this answer
  - [ ] How it was verified
  - [ ] What deep mode changed
- [ ] Add entry transitions for new cards.
- [ ] Keep auto-scroll bounded and non-jarring.
- [ ] Keep the surface readable without overpowering the orb.

## Phase 11 - Input Dock and Task Controls
- [x] Build a premium bottom input dock.
- [ ] Add:
  - [ ] mic/voice button
  - [ ] text input
  - [ ] send button
  - [ ] thinking-time slider
  - [ ] quick mode buttons
  - [ ] stop button during active runs
  - [ ] pause/resume for long-horizon runs
- [ ] Add quick mode buttons:
  - [ ] Fast
  - [ ] Deep
  - [ ] Long Horizon
- [ ] Add optional compact toggles:
  - [ ] Web Allowed
  - [ ] Verification Priority
  - [ ] Local Only
  - [ ] Capability Session Enabled
  - [ ] Cloud Helper Allowed
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
- [ ] Move compressor insights into the drawer.
- [ ] Move optimizer suggestions into the drawer.
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
- [ ] Build a dedicated long-horizon session tray.
- [ ] Show:
  - [ ] elapsed time
  - [ ] current cycle type
  - [ ] checkpoint count
  - [ ] candidate growth
  - [ ] verification passes
  - [ ] evidence refresh count
  - [ ] confidence improvement trend
  - [ ] advisory suggestions considered
  - [ ] early-stop reason
- [ ] Add pause/resume/cancel controls.
- [ ] Add a duty-cycle visualization.
- [ ] Add resource-throttle notices when long-horizon work is intentionally slowed.
- [ ] Add "what extra time bought" summaries.

## Phase 16 - Local-AI Control Plane
- [ ] Turn the existing model and routing surface into a flagship panel.
- [ ] Show:
  - [ ] installed local roles
  - [ ] active routed roles
  - [ ] current routing decision
  - [ ] heavy-slot usage
  - [ ] sidecar helpers
  - [ ] fallback reasons
  - [ ] readiness gaps
  - [ ] route history
  - [ ] optimizer suggestions relevant to routing
- [ ] Add quick actions:
  - [ ] warm model
  - [ ] unload model
  - [ ] enable role
  - [ ] disable role
  - [ ] test ping
  - [ ] inspect fallback reason
  - [ ] open readiness guidance
- [ ] Add structured route summaries to the active task card and answer card where relevant.

## Phase 17 - Capability and Session UX
- [ ] Surface compact capability context in the main shell:
  - [ ] active session badge
  - [ ] target app/window badge
  - [ ] pending approval pill
  - [ ] observation tier badge
  - [ ] last action preview
  - [ ] policy state summary
- [ ] Keep full capability registry and audit trail in secondary views.
- [ ] Build shell-consistent approval prompts.
- [ ] Make approval prompts clearly explain:
  - [ ] what action is requested
  - [ ] why approval is needed
  - [ ] what the target is
  - [ ] what the risk level is
- [ ] Keep approval state visible in the orb and activity strip.

## Phase 18 - Secondary Sheets and Views
- [x] Build lower sheets or modal views for:
  - [x] task history
  - [ ] run inspector
  - [x] knowledge library
  - [x] settings
  - [x] readiness/preflight
  - [x] capability details
  - [x] debug/event log
- [ ] Restyle existing surfaces to match the new shell without removing functionality.
- [ ] Keep export and import actions reachable.
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
  - [ ] orb size
  - [ ] animation intensity
  - [ ] reduced motion
  - [ ] ambient reactivity
  - [ ] particle density
  - [ ] side drawer defaults
  - [ ] activity strip visibility
  - [ ] task timeline visibility
  - [ ] resource ribbon visibility
  - [ ] notification visibility
- [ ] Add persisted performance settings:
  - [ ] low-resource mode
  - [ ] reduced-effects mode
  - [ ] animation frame cap
  - [ ] simplified orb mode
- [ ] Add presets:
  - [ ] Minimal
  - [ ] Balanced
  - [ ] Immersive
- [ ] Add accessibility controls:
  - [ ] low motion
  - [ ] higher contrast
  - [ ] larger status text
  - [ ] simpler accent behavior
- [ ] Keep settings import/export compatible with existing profiles.

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
- [ ] Update packaged app entrypoints so the PySide6 shell becomes the default UI.
- [ ] Keep source-from-repo workflow valid.
- [ ] Ensure first-run onboarding matches the new shell language and layout.
- [ ] Ensure preflight/readiness can be opened from the new shell.
- [ ] Ensure startup failures still degrade safely to stub mode with readable guidance.
- [ ] Ensure diagnostics and support bundle export remain reachable.
- [ ] Ensure packaged build includes the required PySide6 runtime pieces.

## Phase 23 - Regression and Test Coverage
- [ ] Keep all current backend tests green.
- [ ] Add shell-state mapping tests for:
  - [ ] idle
  - [ ] listening
  - [ ] planning
  - [ ] local retrieval
  - [ ] web retrieval
  - [ ] fast reasoning
  - [ ] deep reasoning
  - [ ] critic verification
  - [ ] compression
  - [ ] responding
  - [ ] speaking
  - [ ] degraded mode
  - [ ] approval pending
  - [ ] error
  - [ ] offline
- [x] Add PySide6 widget tests for:
  - [x] shell launch
  - [x] shell shutdown
  - [ ] orb demo states
  - [ ] drawer open/close
  - [ ] timeline population
  - [x] conversation/task cards
  - [x] resource ribbon
  - [ ] settings persistence
  - [ ] reduced-effects mode
  - [ ] low-resource mode
  - [ ] resize behavior
- [ ] Add flow tests for:
  - [ ] stub-mode launch
  - [ ] real-mode launch
  - [ ] fast task
  - [ ] deep task
  - [ ] long-horizon task
  - [ ] degraded/fallback path
  - [ ] web fallback
  - [ ] critic rejection
  - [ ] compressor activity
  - [ ] optimizer advisory visibility
  - [ ] task history access
  - [ ] knowledge library access
  - [ ] settings/profile round-trip
  - [ ] readiness/preflight flow
  - [ ] local-AI control plane flow
  - [ ] capability session visibility
  - [ ] approval flow
  - [ ] packaged Windows launch
  - [ ] event burst throttling
- [ ] Run Qt tests in offscreen mode for CI stability.

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
