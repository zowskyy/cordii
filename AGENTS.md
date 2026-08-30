# AGENTS — Cordi v2

Instructions for AI coding assistants operating in this repo. **Loaded on every session; must be read before writing any file.**

## Pre-Write Methodology (first principles — apply before any write)

Work only within these hard axioms. Decompose every change to one or more of them; verify each claim via `file:line` or execution — never assume.

- **1.5B ceiling.** `qwen2.5-coder:1.5b` is the budget target: it is context-starved and flaky. Design downward from this, not upward. Capacity math (`scripts/capacity_calculator.py:12-20`): `tokens = guidance + base_overhead + N*per_file + folds*delta`; a 1.5B pass needs ~1.3 rounds/file vs 1.05 for 7B+ (`scripts/capacity_calculator.py:15`).
- **4k ctx.** 4096-token window (1.5b preset); the per-model pruner budget and message cap come from the calibration table (`core/context.py:28-31`), applied in `plugins/agent/loop.py` `start()`; the 1.5b budget = 3000 leaves 1k KV headroom (core fallback default `core/context_pruner.py:26`). Folds trigger when `tokens > pruner_budget OR messages > max_messages` (`plugins/agent/loop.py:378`).
- **Sandbox.** No untrusted execution. Every tool/dispatch route is the registered plugin path through `core/registry.py`; there is no backdoor.
- **Zero-token guarantee.** Any new router/dispatch must be deterministic (regex/sympy), adding zero LLM tokens in `lite` profile. Never route a query through an LLM unless gated behind `--enable-semantic-router` and only in `full` profile — applies to the semantic router (`plugins/agent/loop.py:314-319`, `plugins/agent/semantic_router.py:37-38,54`, `main.py:108,112`) AND the multi-domain unresolved-fragment fallback (`plugins/agent/loop.py:259-267`).
- **Pool philosophy.** Lite = 19 plugins (default, zero-token) (`main.py:83-106`); Full = 42, but SemanticRouter stays OFF unless explicitly enabled (`main.py:108-112`). A default-ON LLM step is a P0 regression.
- **Single-pruner preservation.** `assistant` messages with `tool_calls` MUST be preserved for 1.5B coherence (`core/context_pruner.py:26`, `core/summarizer.py:64-65`); never let a second pruner delete reasoning (`plugins/agent/loop.py:374-381`).
- **Injection hardening.** Prompt/context injections are injected as `user` with the `[injected context]` prefix, NEVER `system` (`plugins/agent/loop.py:411-417`).
- **Event hygiene.** Exactly one `turn.start` (outer) + `turn.round` per iteration; no duplicate emits (`plugins/agent/loop.py:302,404`).
- **Snapshot integrity.** Event log snapshots are `base64+zlib` (`core/event_log.py:5,91,95,110`); cache `entries` must be nested to avoid a `"version"` key collision (`plugins/agent/semantic_router.py:171-186`, `plugins/model/embedding.py:118-133`).
- **Calibration separation.** Model-specific numbers (token budget, message cap, rounds/file, safety) live ONLY in the calibration table `core/context.py:28-31` and flow through `Context.config["calibration"]` (`main.py:67-74`); invariant layers read them via `calibration_from_context()` and never as literals. Scaling to a larger model = re-measuring the table (`scripts/capacity_calculator.py --verify`, `--live` benchmark pool), not a code change.

## Implementation Scope (how to make changes — read before editing)

Only two paths are permitted for ANY code addition/change:

### A. Edit/extend existing core files only (default)
Reuse the pre-existing files through their public surface — `core/*` (`plugin.py`, `registry.py`, `context.py`, `context_pruner.py`, `summarizer.py`, `messages.py`, `errors.py`, `event_log.py`, …), `plugins/*` (existing plugins), `scripts/*`, `tests/*`, `main.py`, `.gitignore`. No new file trees.

### B. New plugin/module (only if it STILL uses pre-existing files + routing)
A brand-new plugin is allowed ONLY if ALL hold:
1. It subclasses the existing base at `core/plugin.py:8` (`Plugin` or `EventDrivenPlugin`), inheriting `register`/`start`/`stop`/`on_event`.
2. It declares `name` and `dependencies` (`core/plugin.py:9-11`).
3. It registers through the EXISTING mechanism — `register()`, `register_class()`, or `discover()` (`core/registry.py:24,35,184`), so it participates in dependency-aware topological sort (`core/registry.py:46,223`). It must NOT invent a parallel loader.
4. It is wired into `main.py` under the `lite` block (`main.py:83-106`) or the `full` block (`main.py:108+`); a plugin that exists but is never registered is dead code.
5. It carries its weight: zero-token in `lite` (deterministic), LLM-only in `full` and gated, per the zero-token guarantee above.
6. Its tests live alongside existing ones in `tests/` (`tests/test_*.py`) and follow the existing pytest style.

### Forbidden
No standalone scripts, no new package roots, no bypassing `Plugin`/`Registry`, no new event buses, no new config files outside the existing `Context.config` shape (`main.py:67-74`).

### Ratified auxiliary trees (documented exceptions, all pre-date the two-path rule)
- `benchmark/` — R&D/eval track: task pools, verifiers, canary, data pipeline, finetune. Consumes the pool (imports `core/*`/`plugins/*`) but never registers plugins into the product path. One-way dependency: `core/` and `plugins/` must NOT import from `benchmark/`. Heavy finetune deps (torch/transformers/peft/trl/datasets) are opt-in and intentionally NOT in `requirements.txt`.
- `ui.py` — local Gradio test harness over the pool (documented in README quick-start).
- `plugins/model/ensemble.py` — eval-track plugin used by `benchmark/canary.py`; deliberately NOT in the lite/full pool.
- `tests/failure_recovery/debug_benchmark_tasks.py` — fault-injection debug runner for the failure-recovery test suite (run from repo root).

## Verification Gate (before any "complete" claim)
- `pytest --basetemp C:\tmp\pytest_cordiiv2` must show 248+ passing.
- `.gitignore` coverage: `git check-ignore` must confirm `.cache/*`, `continuity/*.db`, `*.db`, `workspace/*` and cache nesting are ignored (`.gitignore:9-14`).


