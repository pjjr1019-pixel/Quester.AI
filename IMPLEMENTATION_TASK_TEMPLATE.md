# Implementation Task Template

Use this brief at the start of every implementation pass. The goal is to keep
Phase 13 guardrails enforceable in normal repo workflow instead of leaving them
as checklist text only.

## Standing Phase 13 Checklist

- Review the current phase before coding and confirm the prior phase acceptance
  checks are already green.
- Keep the change scoped to one subsystem and one primary test surface where
  possible.
- Name the target files, typed contracts, and success tests up front so the
  implementer does not need to guess.
- Preserve both `stub_mode=true` and `stub_mode=false` behavior when touching
  runtime or readiness paths.
- Keep typed parsing and validation at every agent boundary.
- Allow one structured-output repair attempt only; after that, return the
  deterministic or structured fallback path instead of retrying indefinitely.
- Call out any feature-parity risk explicitly before simplifying internals.

## Task Brief

- Phase:
- Prior acceptance gate already green:
- Target subsystem:
- Target files:
- Required contracts or schemas:
- Behavior change:
- Primary tests to add or update:
- Secondary validation commands:
- Modes touched:
  - `stub_mode=true`
  - `stub_mode=false`
  - `headless`
- Feature-parity risks:
- Expected completion signal:
