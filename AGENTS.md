# Dice Chess Training — AI Agent Guidelines

## Architecture Overview

- **Domain**: Open training pipeline for Dice Chess: a GPU **label factory** (Star2-pruned depth-3 tree expansion on CPU, batched leaf evaluation and exact 216-roll rescoring on GPU) and two small networks trained on its labels, designed to scale from a single workstation to an HPC cluster. Public counterpart of the private `dicechess-ev` pipeline (open-core split: the framework is open; production-tuned weights, opening books, and tournament bot configurations stay private).
- **Planned stack**: Python >= 3.12 with `uv` (PyTorch, ONNX; LightGBM where useful) for training; the JVM `dicechess-engine` (Scala 3) for tree expansion and self-play workers; `mise` as the task runner.
- **Planned workflow**:
  1. Position sampling: archived games plus self-play from headless engine workers, deduplicated and phase-stratified.
  2. Label factory: CPU workers expand Star2-pruned depth-3 search trees; GPUs batch leaf evaluations and exact 216-roll candidate rescores, emitting depth-3 value labels and per-candidate rescore distributions.
  3. Training: a chance-collapse afterstate value net and a listwise move pre-ranker (PyTorch; DDP for sweeps).
  4. Export: INT8 ONNX for the engine's chance-node hook and phase-1 pre-ranker slot.
  5. Evaluation: holdout agreement with the depth-3 teacher (MSE, rank correlation, log-loss/calibration), fixed-time arena A/B matches, and rated games on the public bot ladder.

The authoritative program description lives in `README.md`; keep this overview in sync with it.

## Status

**Scaffold — no code yet.** The pipeline is being built ahead of the European AI Hackathon (October 2026); the training and distillation parts are extracted from the private `dicechess-ev` repository, the label factory is new code on top of the open engine. Until `mise.toml` and `pyproject.toml` land, there are no build, test, or lint tasks to run. Once tooling lands, this section is replaced by the standard Fortemate developer workflow (`mise run setup` / `check` / `test` / `format`).

## Branch & Issue Guidelines

- **Branches**: `feat/[ID]-[description]`, `fix/[ID]-[description]`, `chore/[ID]-[description]`, `docs/[ID]-[description]`.
- **GitHub Issues**: Use native GitHub Issue Types (`Feature`, `Task`, `Bug`) rather than issue labels.
- **PR Description**: Reference closed issues with `Closes #ID`.

## Issue management

<!-- dc-shared:issue-management v5 — keep identical across Fortemate repositories -->

- Use the native GitHub Issue Type as the canonical work classification:
  - `Bug` for unexpected or incorrect behavior.
  - `Feature` for a request, idea, or new user-visible capability.
  - `Task` for a specific piece of engineering, research, maintenance, or documentation work.
- Never commit directly to a repository's default branch. Name branches `<type>/<short-description>` or `<type>/<id>-<short-description>` using the canonical types `task|feat|bug|refactor|chore|docs|ci|test|perf`. Include an Issue id only when the pull request is intended to fully complete that Issue; otherwise omit it or use the id of an independently actionable sub-issue. Example: `bug/42-fix-dfen-parser`.
- Do not apply `bug` or `enhancement` labels to Issues merely to repeat their Type. Keep those labels for pull-request release classification. On Issues, labels describe only a technical domain or cross-cutting concern, and only existing repository labels may be used.
- Applying the `jules` label is a live execution trigger. On an open Issue the label denotes the current Jules delegation; on a closed Issue it may remain as historical execution metadata. By default, agents must never apply or remove it. Exception: a top-level Codex or Claude Code orchestrator directly handling the current human request may apply or remove `jules` only when that human explicitly authorizes Jules delegation for the current parent task. Jules, Antigravity, CI, delegated subagents, and agents without that task-scoped authorization must never dispatch or recursively delegate work.
- Before an authorized orchestrator applies `jules`, it must read the Issue back and verify that it is an open, independently mergeable leaf Issue with no blocker, competing owner or pull request, overlapping active work, or dependency on unmerged changes; belongs to Fortemate Engineering; has Status `Ready`, Execution tier `Routine`, and `spec:ready`; and contains self-contained Context, Objective, testable Definition of Done, Guards, Verification gates, Non-goals, and a bounded file-level blast radius. Apply `jules` last, read it back, monitor the Issue/session/pull request through completion, review the result, and take over stalled work. Never dispatch the same task through both the label and Jules CLI. Follow the `jules-delegation` skill when it is available.
- Actionable Jules feedback must be a submitted pull-request conversation or inline comment from the GitHub user who triggered the task, explicitly mention `@jules`, and be followed by acknowledgement and re-review of the resulting commit. A review body is not a Jules feedback channel. A delegated pull request and its commits may close only its leaf Issue, never its parent or sibling.
- Removing `jules` or using Jules CLI pull/teleport does not prove that the remote session stopped. Never write concurrently to a possibly active Jules branch. Continue the existing pull request only after terminal state is confirmed; otherwise recover verified work in an isolated branch and replacement pull request.
- After successful Jules work closes an Issue, retain `jules` as an audit marker. If that Issue is reopened, remove the historical label before triage; reapply it only after fresh task-scoped authorization and all dispatch checks, because the new label event starts a new session. During takeover of an open Issue, remove `jules` and record `outcome:escalated`.
- Before creating or updating an Issue, search relevant Fortemate repositories across open and closed Issues for semantic duplicates. Read the live Types, field options, labels, assignees, and relationships before mutation; never rely on cached IDs or invent metadata.
- GitHub-facing work items are English-only. Use the appropriate Issue Form when available, or `gh issue create --body-file <file>` for CLI creation; never pass a multiline body inline. Every Issue must contain `Context`, `Objective`, and a testable `Definition of Done`.
- Add every actionable Issue (never pull requests) to the organization Project [Fortemate Engineering](https://github.com/orgs/fortemate/projects/1).
- Use Project `Status` only for workflow state:
  - `Backlog` means triaged but not committed for active work.
  - `Ready` means sufficiently defined and available to start.
  - `In progress` means someone is actively working on it.
  - `In review` means implementation is waiting for review or validation.
  - `Done` means the Issue is closed.
- Set the Project `Execution tier` during triage:
  - `Routine` for a bounded, reversible task suitable for Jules or another low-cost agent.
  - `Mid` for a well-scoped task that needs a stronger coding agent with iterative supervision.
  - `Frontier` for architecture, public contracts, complex diagnosis, or other high-blast-radius work; human-led.
  - `Human-only` for releases, production operations, secrets, or legal decisions that must never be delegated.
  - `Decompose` for work too large to route as-is: split it into sub-issues, tier each, then re-tier or close the parent.
  - A blank value means the Issue has not been routed yet.
- Leave the organization `Priority` Issue field blank for normal work. Set it only to deliberately jump the queue: `Urgent` for an immediate incident, security problem, or release blocker; `High` for important or blocking planned work. Never replace organization fields with labels or duplicate Project fields.
- Triage establishes Type, Execution tier, applicable labels, Project membership, Status, and relationships (plus Priority only for queue-jumpers). Assign an Issue only when a person owns its next action, and assign the active owner before moving it to `In progress`; unassigned means agent pool or no current owner, not low priority.
- Use parent/sub-issue relationships for independently actionable decomposition, `Blocking`/`Blocked by` for hard ordering dependencies, and `Relates to` for non-blocking associations. If the live UI or API cannot create a relation, add an explicit typed cross-reference that preserves its semantics: `Parent:`, `Sub-issue:`, `Blocking:`, `Blocked by:`, or `Related:` followed by `owner/repository#<id>`. Do not simulate relationships with title prefixes, labels, or duplicate task lists.
- When a pull request targets the repository's default branch and fully completes an Issue, link it with `Closes #<id>` or `Closes owner/repository#<id>`. Use a non-closing reference for partial work or for a pull request targeting any other branch.
- After every Issue, pull-request, or Project mutation, read the item back. For an Issue, verify Type, Issue fields, labels, assignee, relationships, Project membership, and Status. For a pull request, verify base/head branches, draft and merge state, labels, assignees/reviewers, and linked Issues; pull requests are never Project items, and Issue Type and Issue fields do not apply. Report any metadata that the available API or UI could not set.
- The human owner reviews, approves, and merges pull requests. Agents never merge pull requests or execute releases.

<!-- /dc-shared:issue-management -->
