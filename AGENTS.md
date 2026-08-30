
```markdown
# Cordi v2 Agent Instructions

> **Mandatory repository instructions for coding agents.**
>
> Read this file at session start and apply it before planning, editing, creating, deleting, moving, running, or claiming completion of any work.
>
> If a user request conflicts with this file, preserve the hard invariants below unless the user explicitly authorizes changing the relevant invariant and its tests.

---

## 0. Core Operating Rule

Work from evidence, not assumptions.

Before modifying a behavior:

1. Inspect the relevant implementation, tests, and configuration.
2. Identify the invariant(s) affected.
3. Write or update a focused failing test first.
4. Make the minimum implementation change that makes the test pass.
5. Refactor only while tests are green.
6. Run the required verification before claiming completion.

Use `file:line` references or command output to support claims. Do not invent file contents, APIs, configuration values, test outcomes, repository structure, or ignore behavior.

If evidence is insufficient, inspect first. If the change is high-impact or crosses an Ask-First boundary, stop and ask.

---

## 1. Project Mission

Cordi v2 is a multi-profile agent/plugin system designed for reliable operation under constrained local models.

The primary design target is:

```text
Model: qwen2.5-coder:1.5b
Context window: 4096 tokens
Working budget: 3000 tokens
Reserved KV headroom: 1000 tokens
```

The repository must remain reliable for the 1.5B target first. Larger models may improve performance, but they must not be required for correctness, architecture, routing, profile safety, or normal `lite` operation.

**Design downward from 1.5B; never upward from a larger model.**

---

## 2. Non-Negotiable Invariants

Every change must preserve these invariants unless the task explicitly changes the invariant, its verification, and its documentation together.

### 2.1 1.5B Capacity Ceiling

`qwen2.5-coder:1.5b` is the budget model.

Capacity model:

```text
tokens = guidance + base_overhead + N * per_file + folds * delta
```

Canonical implementation:

```text
scripts/capacity_calculator.py:12–20
```

Known calibration relationship:

```text
1.5B target: approximately 1.3 rounds/file
7B+ target: approximately 1.05 rounds/file
```

Do not add unnecessary static instructions, duplicated architecture explanations, broad tool output, large inline code copies, or always-on agent steps that make the 1.5B profile less reliable.

---

### 2.2 4k Context Window

The 1.5B preset has a hard 4096-token context limit.

Rules:

- The working pruner budget for 1.5B is **3000 tokens**.
- The remaining approximately 1000 tokens are KV/cache headroom.
- Per-model token budgets, message caps, rounds-per-file values, tool-result byte caps, and safety values belong only in the calibration table.
- The calibration table is defined in `core/context.py:28–34`.
- Calibration is injected through `Context.config["calibration"]` in `main.py:67–74`.
- Invariant layers must read values through `calibration_from_context()`.
- Do not duplicate calibration values as literals in plugin logic, routing logic, pruning logic, or tests unless the test is explicitly validating a default/fallback.

Context folds must trigger when either condition is true:

```text
tokens > pruner_budget
OR
messages > max_messages
```

Relevant loop behavior:

```text
plugins/agent/loop.py:388
```

The pruner must enforce both:

1. A message-count pass.
2. A token-budget pass.

It must remove lowest-score eligible messages until both constraints are satisfied, even if message count alone is under its cap.

Relevant implementation:

```text
core/context_pruner.py
```

A single tool result must be truncated to its calibrated byte cap before storage, so one result cannot consume the full context window.

Relevant implementation:

```text
plugins/agent/loop.py::_record_tool_result
```

---

### 2.3 Zero-Token Guarantee

The default `lite` profile is zero-token.

This is a hard product guarantee, not an optimization preference.

Rules:

- Any new route, dispatch decision, classifier, or fallback used by `lite` must be deterministic.
- Allowed deterministic approaches include explicit rules, regex, parsers, structured matching, and symbolic methods such as SymPy.
- A deterministic system must not make hidden LLM calls, embedding calls, semantic-router calls, model-selection calls, or “one small classification” calls in `lite`.
- New default-on LLM work is a **P0 regression**.
- Never send a query through an LLM unless both conditions are true:
  1. `--enable-semantic-router` is explicitly enabled.
  2. The active profile is `full`.

This applies specifically to:

```text
plugins/agent/loop.py:269–277
plugins/agent/loop.py:324–329
plugins/agent/semantic_router.py:37–38,54
main.py:108,112
```

When modifying routing, prove that `lite` remains zero-token through focused tests and/or instrumentation.

---

### 2.4 Profile Philosophy

Profiles are intentional product boundaries.

```text
lite:
  19 plugins
  Default profile
  Zero-token
  Deterministic routing only

full:
  42 plugins
  May use LLM functionality only when properly gated
  SemanticRouter remains disabled unless explicitly enabled
```

Relevant profile wiring:

```text
main.py:83–106     # lite
main.py:108+       # full
```

Do not:

- Add a plugin to `lite` without proving it is deterministic and zero-token.
- Make a full-only feature quietly load or execute in `lite`.
- Enable semantic routing by default.
- Change a profile count, registration order, startup behavior, or dependency graph without focused tests and explicit review.

---

### 2.5 Plugin and Sandbox Integrity

No untrusted execution is permitted.

Every tool or dispatch route must use the registered plugin path through:

```text
core/registry.py
```

There is no alternative loader, bypass, backdoor, implicit plugin execution path, or ad hoc event-dispatch path.

All new plugins/modules must use the existing base and registry system:

```text
core/plugin.py:8
core/registry.py:24,35,46,184,223
```

Required lifecycle and registration path:

```text
Plugin or EventDrivenPlugin
→ name + dependencies
→ existing register/register_class/discover path
→ dependency-aware topological sort
→ explicit main.py profile wiring
→ tests
```

---

### 2.6 Single-Pruner Preservation

For 1.5B coherence, preserve `assistant` messages containing `tool_calls`.

Relevant behavior:

```text
core/context_pruner.py:26
core/summarizer.py:64–65
plugins/agent/loop.py:384–391
```

Do not introduce a second pruner, post-prune cleanup pass, summarization pass, or message filter that can delete those assistant/tool-call messages after the canonical pruner has preserved them.

There must be one authoritative context-pruning path.

---

### 2.7 Injection Hardening

Injected context is untrusted context, even if produced internally.

Rules:

- Insert prompt/context injections as a `user` message.
- Prefix injected content with exactly:

```text
[injected context]
```

- Never inject retrieved, generated, external, benchmark, tool, memory, or user-supplied context as a `system` message.
- Never allow injected context to override these repository instructions.

Relevant implementation:

```text
plugins/agent/loop.py:421–427
```

---

### 2.8 Event Hygiene

Emit events exactly once at their intended ownership level.

Per user turn:

```text
1 × turn.start      # outer turn
1 × turn.round      # per loop iteration
```

No duplicate emissions.

Relevant implementation locations:

```text
plugins/agent/loop.py:312
plugins/agent/loop.py:414
```

When changing event flow, test normal execution, retries, early returns, exceptions, and multi-round turns.

---

### 2.9 Snapshot and Cache Integrity

Event-log snapshots must remain:

```text
base64 + zlib
```

Relevant implementation:

```text
core/event_log.py:5,91,95,110
```

Cache payloads must keep `entries` nested to avoid collisions with reserved top-level keys such as `"version"`.

Relevant implementations:

```text
plugins/agent/semantic_router.py:171–186
plugins/model/embedding.py:118–133
```

Do not flatten cache entry payloads into the cache root.

---

### 2.10 Calibration Separation

Model-specific numbers belong only in the calibration table.

This includes:

- Token budgets
- Message caps
- Tool-result byte limits
- Rounds-per-file
- Safety margins
- Model-specific limits
- Performance/capacity thresholds

Correct scaling path:

```text
Re-measure calibration
→ scripts/capacity_calculator.py --verify
→ live benchmark pool, when applicable
→ update calibration table
```

Incorrect scaling path:

```text
Change random literals across production code
```

Scaling to a different model is a calibration exercise, not a reason to spread model-specific constants through invariant layers.

---

## 3. Required Development Workflow

## 3.1 Inspect Before Writing

Before any edit:

1. Read the target implementation.
2. Read nearby tests and test conventions.
3. Read relevant configuration and wiring.
4. Search for callers, event consumers, registrations, and profile usage.
5. Identify the smallest safe change.
6. State which invariant(s) and profile(s) are affected.

Do not create speculative abstractions or broad refactors before understanding the local design.

Prefer narrow inspection:

```powershell
Get-Content path\to\file.py
Select-String -Path .\**\*.py -Pattern "symbol_name"
git diff
git status --short
```

Use repository-native commands and the current project environment. Do not assume dependencies, commands, or tools exist unless verified.

---

## 3.2 Mandatory TDD: Red → Green → Refactor

Every behavior change and bug fix follows TDD.

### Red

1. Add one focused test describing one observable behavior.
2. Use a descriptive behavior-oriented name.
3. Run the targeted test.
4. Confirm it fails for the exact missing/broken reason.

Example:

```python
def test_pruner_drops_lowest_score_messages_when_over_token_budget():
    ...
```

If the new test passes before implementation, it does not prove the new behavior. Fix the test, fixture, assertion, or setup before continuing.

Never skip the red phase by writing implementation first and tests afterward.

### Green

1. Make the smallest production change that passes the failing test.
2. Do not combine unrelated cleanup, formatting sweeps, renames, or architecture changes.
3. Run the focused test immediately.
4. Run affected neighboring tests if behavior crosses a module boundary.

### Refactor

1. Refactor only while the targeted test and relevant suite remain green.
2. Preserve external behavior.
3. Remove duplication only when it improves local clarity and does not hide calibration/profile boundaries.
4. Re-run the appropriate test scope after each meaningful refactor.

### Test Quality Rules

Tests must:

- Assert observable behavior, not incidental implementation details.
- Be deterministic.
- Avoid network, model, clock, random, and filesystem dependence unless explicitly controlled.
- Use fixtures/mocks only at external boundaries.
- Cover success, failure, boundary, and regression behavior where relevant.
- Follow existing pytest naming and layout conventions.
- Live in the existing `tests/` structure.

Do not weaken, delete, skip, or mark a test `xfail` merely to obtain a green suite without explicit user authorization.

---

## 4. Permitted Change Paths

Only two paths are permitted for code additions or changes.

## 4.1 Path A — Edit or Extend Existing Files

This is the default.

Reuse existing public surfaces in:

```text
core/*
plugins/*
scripts/*
tests/*
main.py
.gitignore
```

Examples of expected reuse:

```text
core/plugin.py
core/registry.py
core/context.py
core/context_pruner.py
core/summarizer.py
core/messages.py
core/errors.py
core/event_log.py
```

Rules:

- Extend existing architecture before creating another abstraction.
- Reuse the current plugin, registry, event, config, context, and test systems.
- Do not create a new top-level package root or a parallel subsystem.
- Keep changes local to the smallest correct ownership boundary.

---

## 4.2 Path B — New Plugin or Module

A new plugin/module is permitted only when **all** conditions below are true.

1. It subclasses `Plugin` or `EventDrivenPlugin` from `core/plugin.py:8`.
2. It declares `name` and `dependencies` as required by `core/plugin.py:9–11`.
3. It registers through the existing mechanism:
   - `register()`
   - `register_class()`
   - `discover()`
4. It participates in dependency-aware topological sorting through `core/registry.py`.
5. It is explicitly wired into exactly the appropriate `main.py` profile block:
   - `lite`, only if deterministic and zero-token.
   - `full`, if LLM-backed and correctly gated.
6. It has focused pytest coverage in the existing `tests/` layout.
7. It does not create a parallel loader, event bus, config format, dispatch route, lifecycle, or plugin hierarchy.
8. It carries its weight:
   - `lite`: deterministic, zero-token, justified.
   - `full`: LLM-only behavior is explicitly gated.

An unregistered plugin is dead code. Do not add one.

---

## 5. Forbidden Changes

Do not do any of the following unless the user explicitly asks to change this policy and the full corresponding architecture/test impact is addressed:

- Create a standalone production script that bypasses the plugin/registry system.
- Create a new top-level package root or file tree.
- Bypass `Plugin`, `EventDrivenPlugin`, or `Registry`.
- Create a parallel plugin loader.
- Create a new event bus or duplicate event ownership.
- Create a new configuration file/schema outside the established `Context.config` shape.
- Add LLM work to `lite`.
- Turn on SemanticRouter by default.
- Add model-specific literals outside calibration.
- Add a second pruning path that can remove preserved tool-call messages.
- Insert untrusted context as `system`.
- Import `benchmark/` from `core/` or `plugins/`.
- Commit generated runtime state, caches, databases, IDE settings, or benchmark result artifacts.
- Delete tests, weaken assertions, or lower thresholds simply to make CI pass.
- Claim completion without running the applicable verification commands.

---

## 6. Repository Structure and Ownership

```text
core/             Core invariants: plugin API, registry, context, pruning, messages, errors, events
plugins/          Product plugins registered through the existing Plugin/Registry system
scripts/          Maintained development, calibration, and capacity tooling
tests/            pytest suite, including failure-recovery tests
benchmark/        R&D/evaluation track; one-way dependency boundary
follow/           Unclassified local directory; inspect before use or modification
continuity/       Runtime continuity/state directory; only .gitkeep is tracked
workspace/        Runtime workspace directory; only .gitkeep is tracked
cache/            Visible directory with unverified ownership; do not assume it is ignored
calib_test_tmp/   Ignored calibration/test temporary output
main.py           Profile wiring and calibration configuration
ui.py             Local Gradio test harness
.gitignore        Source of truth for ignored/generated/runtime paths
```

---

## 7. Auxiliary Tree Rules

## 7.1 `benchmark/`

`benchmark/` is a ratified R&D/evaluation tree.

It may contain task pools, verifiers, canary checks, data pipelines, and finetuning-related code.

Rules:

- `benchmark/` may import from `core/` and `plugins/`.
- `core/` and `plugins/` must never import from `benchmark/`.
- Benchmark code must not register production plugins into the product path.
- Heavy optional finetuning dependencies such as `torch`, `transformers`, `peft`, `trl`, and `datasets` remain opt-in and must not be added to ordinary product requirements unless explicitly authorized.

This is a strict one-way dependency:

```text
benchmark/  →  core/
benchmark/  →  plugins/

core/       ✗  benchmark/
plugins/    ✗  benchmark/
```

---

## 7.2 `ui.py`

`ui.py` is a local Gradio test harness over the pool.

Do not turn it into a second application architecture, alternate plugin loader, or source of production profile behavior.

---

## 7.3 `plugins/model/ensemble.py`

`plugins/model/ensemble.py` is eval-track functionality used by `benchmark/canary.py`.

It is intentionally not part of the standard `lite` or `full` pool.

Do not add it to normal profile registration unless explicitly authorized and fully re-evaluated.

---

## 7.4 Failure-Recovery Debug Runner

```text
tests/failure_recovery/debug_benchmark_tasks.py
```

This is a fault-injection debug runner for failure-recovery tests. Run it from repository root when relevant.

Do not convert it into a general product entrypoint.

---

## 8. Git and Generated-State Rules

The actual `.gitignore` is authoritative. Never infer ignore behavior from a directory name alone.

### 8.1 Tracked Placeholder Files

The following placeholder files preserve otherwise runtime-managed directories:

```text
continuity/.gitkeep
workspace/.gitkeep
```

All other contents of those runtime directories are ignored:

```text
continuity/*
workspace/*
```

with the tracked `.gitkeep` exceptions.

Do not delete those placeholders unless the directory-management policy is intentionally changing.

---

### 8.2 Known Ignored Paths

Do not edit as source, commit, or rely on these as durable repository artifacts unless the task explicitly concerns ignore rules, cleanup behavior, or a controlled test fixture:

```text
.venv/
.idea/
calib_test_tmp/
__pycache__/
.pytest_cache/
.pytest_*/
.pytest-*/
*.pyc
*.db
*.db-journal
.cache/*
continuity/*
workspace/*
benchmark_results.db
benchmark_results.json
```

Known exceptions:

```text
.cache/.gitkeep
continuity/.gitkeep
workspace/.gitkeep
```

---

### 8.3 `cache/` Is Not `.cache/`

The repository visibly contains `cache/`, while the shown ignore rule applies to `.cache/*`.

These are different paths.

Until verified, treat `cache/` as **unclassified**:

- Do not assume it is ignored.
- Do not assume it is generated.
- Do not delete it.
- Do not add it to `.gitignore` automatically.
- Do not import from it or treat it as source without inspection.

Before modifying its policy, inspect it:

```powershell
Get-ChildItem -Force cache
git status --ignored --short cache
git check-ignore -v cache\*
git ls-files cache
```

If it is intentional generated state, add an ignore rule only after verifying there is no required tracked content.

---

### 8.4 `follow/` Is Unclassified

`follow/` was not shown as tracked by `git ls-files` and was not shown in `.gitignore`.

Treat it as unknown/local until inspected.

Before writing, deleting, importing from, or depending on `follow/`:

```powershell
Get-ChildItem -Force follow
git status --ignored --short follow
git check-ignore -v follow\*
git ls-files follow
```

Do not silently classify it as source, fixture data, cache, or disposable output.

---

## 9. Ask-First Boundaries

Stop and ask for confirmation before proceeding with any of the following:

- Changing calibration table values or their interpretation.
- Changing the 1.5B token budget, message cap, tool-result cap, safety margin, or capacity assumptions.
- Adding/removing/reclassifying a plugin in `lite` or `full`.
- Changing profile wiring, startup order, registry ordering, or dependency edges in `main.py`.
- Enabling SemanticRouter by default or changing its gating behavior.
- Changing event ownership, event names, or event emission count.
- Changing pruning policy, summarization policy, or preservation rules for assistant/tool-call messages.
- Changing injection message roles or prefix behavior.
- Importing benchmark code into product code, or moving product code into benchmark dependencies.
- Changing `.gitignore` rules for `cache/`, `follow/`, `continuity/`, `workspace/`, databases, or calibration output without confirming intended ownership.
- Deleting, renaming, or moving directories whose ownership has not been verified.
- Adding dependencies, changing Python/runtime version requirements, or modifying package/install configuration.
- Reducing test coverage, skipping tests, lowering expected pass thresholds, or changing tests solely to fit a new implementation.

---

## 10. Completion and Verification Gate

Never claim a task is complete until all relevant checks pass.

### 10.1 Minimum Required Full Test Gate

Run from repository root:

```powershell
pytest --basetemp C:\tmp\pytest_cordiiv2
```

The result must show:

```text
248+ passing
```

If the project’s verified current baseline has legitimately increased, use the higher actual number. Do not lower the requirement.

---

### 10.2 Required Ignore Verification

When modifying ignore rules, runtime state, generated output, cache behavior, databases, or directory ownership, verify actual `.gitignore` matching:

```powershell
git check-ignore -v .idea\test
git check-ignore -v calib_test_tmp\test
git check-ignore -v __pycache__\test.pyc
git check-ignore -v .pytest_cache\test
git check-ignore -v .cache\test
git check-ignore -v continuity\test.db
git check-ignore -v workspace\test
git check-ignore -v benchmark_results.db
git check-ignore -v benchmark_results.json
```

Verify tracked placeholders remain tracked:

```powershell
git ls-files continuity/.gitkeep workspace/.gitkeep
```

For an unclassified visible directory, inspect before prescribing rules:

```powershell
git status --ignored --short cache
git check-ignore -v cache\*
git ls-files cache

git status --ignored --short follow
git check-ignore -v follow\*
git ls-files follow
```

---

### 10.3 Required Change Review

Before final response, run:

```powershell
git status --short
git diff --check
git diff
```

Confirm:

- Only intended files changed.
- No generated/cache/database/IDE artifact is staged or newly tracked.
- No unrelated formatting churn exists.
- New code has corresponding focused tests.
- Profile behavior remains correct.
- Lite remains zero-token where applicable.
- All relevant test scopes pass.
- Full suite meets the required pass threshold.

---

## 11. Required Final Report Format

When reporting work, provide concise evidence in this order:

1. **What changed** — files and behavior.
2. **Why** — invariant, bug, feature requirement, or test failure addressed.
3. **TDD evidence** — failing test added/updated, then passing result.
4. **Verification** — commands run and outcomes.
5. **Risk/profile impact** — `lite`, `full`, calibration, routing, pruning, events, cache, or injection impact.
6. **Remaining limitations** — only if verified and relevant.

Do not claim tests passed unless you executed them in the current workspace and observed the result.

Do not claim behavior is unchanged unless inspected and/or tested.

---

## 12. Per-Task Decision Checklist

Before any write, answer these internally from repository evidence:

1. What observable behavior is changing?
2. Which core invariant(s) does it affect?
3. Is the active impact in `lite`, `full`, benchmark-only, or multiple paths?
4. Does it risk adding LLM tokens in `lite`?
5. Does it involve calibration values or hardcoded model-specific numbers?
6. Does it affect routing, dispatch, plugins, registry order, events, pruning, summaries, tool calls, injection, snapshots, or cache serialization?
7. Is Path A sufficient, or does Path B satisfy every plugin/module condition?
8. What focused test will fail before implementation?
9. What exact command will verify the implementation?
10. Does it cross an Ask-First boundary?
11. Are any affected directories generated, ignored, runtime-managed, or unclassified?
12. What full-suite verification is required before completion?

If these cannot be answered from evidence, inspect more or ask.

---

## 13. Fast Rules for Small Models

When working with constrained agents, keep the task narrow.

- Read only files relevant to the requested change.
- Prefer targeted searches over broad repository dumps.
- Do not paste huge logs into agent context; summarize evidence with paths and key lines.
- Keep one behavioral objective per implementation cycle.
- Prefer existing utilities and public interfaces.
- Keep patches minimal.
- Run the smallest relevant test first, then broaden verification.
- Avoid speculative refactors.
- Use `file:line` anchors rather than copying large code blocks.
- Treat every default-on LLM step as a potential P0 regression.
- Treat every new literal model limit as a calibration-separation violation until proven otherwise.

---
Cordi v2**
```