# Cordi v2 — 1.5B Usage Guide

This guide explains how to use the Cordi v2 harness effectively with `qwen2.5-coder:1.5b` (and similar 1.5B-class models). It covers recommended workflows, when to use each profile, and how to interpret pruner/fold behavior.

## Quick start

```powershell
# Default 1.5B-safe lite profile
python main.py --workspace workspace --model qwen2.5-coder:1.5b --profile lite

# Dry-run check (CI or quick sanity)
python main.py --model qwen2.5-coder:1.5b --profile lite --dry-run

# Full profile with semantic router (debug/analysis, higher token cost)
python main.py --model qwen2.5-coder:1.5b --profile full --enable-semantic-router
```

## Profiles: when to use which

### `lite` (default, recommended for 1.5B)

**Use for:** everyday coding tasks, file edits, small refactors, tool-driven workflows.

**Characteristics:**
- **21 plugins** total (19 explicit + 2 auto-added by `EventLogger`: `event_log`, `continuity`)
- Zero-token pool: math, datetime, units routing are deterministic/symbolic
- No semantic router, no embeddings, no heavy observability
- Optimized for ~3k token budget within 4k context

**Best practices:**
- Keep file content concise; prefer TEMPLATE-style summaries over full-file dumps when possible.
- Let the deterministic pool handle exact work (math, units, datetime) instead of asking the model to compute.
- Use `--dry-run` before long sessions to confirm plugin counts and profile state.

### `full` (debug/analysis, higher token cost)

**Use for:** deep debugging, tracing, memory experiments, semantic search across workspace.

**Characteristics:**
- **44 plugins** total (42 explicit + 2 auto-added)
- Semantic router available (but **off by default**; enable with `--enable-semantic-router`)
- Observability plugins: tracing, metrics, telemetry, health monitoring
- Higher token consumption; **not recommended** for routine 1.5B work

**Best practices:**
- Enable semantic router only when you need embedding-based retrieval; it adds token and compute cost.
- Monitor token usage closely; expect more frequent folds/prunes.
- Use `full` for targeted sessions, not as your default 1.5B workflow.

## Capacity and context hygiene

### Expected capacity

For `qwen2.5-coder:1.5b` with default calibration:

```powershell
python scripts/capacity_calculator.py --model 1.5b --quiet
# Expected: 19
```

This means you can safely work with **~19 files** in context before the pruner must fold. If you exceed this, expect history compaction.

### Interpreting folds and pruner logs

The pruner triggers a fold when:

- **Tokens** exceed the pruner budget (~3k for 1.5B), **OR**
- **Messages** exceed the model's `max_messages` limit

When a fold occurs:

- Tokens are reduced to ~40% of their previous value plus a small delta
- Messages are reduced to ~60% plus one
- Older turns are compacted; recent turns are preserved

**Signs you're hitting folds too often:**
- Model loses track of earlier instructions
- Repeated re-explanation of context
- Tool outputs get truncated unexpectedly

**Mitigations:**
- Reduce per-file token usage (shorter templates, summaries)
- Work in smaller batches (fewer files per session)
- Use `full` profile sparingly; it consumes tokens faster

## Recommended workflows

### 1. File edit / refactor session

```powershell
python main.py --workspace workspace --model qwen2.5-coder:1.5b --profile lite
```

- Open only the files you need; avoid dumping entire directories.
- Let the model generate edits via tools; review before applying.
- If context feels tight, close unused files or start a fresh session.

### 2. Math / units / datetime tasks

- Ask the model to use the math/units/datetime tools explicitly.
- Example: "Calculate the total bytes for these 5 files using the math tool."
- The deterministic pool handles exact computation; the model focuses on structure.

### 3. Debugging / tracing

```powershell
python main.py --workspace workspace --model qwen2.5-coder:1.5b --profile full --enable-semantic-router
```

- Use tracing, metrics, and health plugins to inspect behavior.
- Enable semantic router only if you need embedding-based retrieval.
- Keep sessions short; switch back to `lite` for routine work.

### 4. CI / pre-merge checks

```powershell
# Verify baseline invariants
powershell -File scripts/baseline_gate.ps1

# Quick profile check
python main.py --profile lite --dry-run
python main.py --profile full --dry-run
```

Run these before merging changes that affect plugins, calibration, or routing.

## Troubleshooting

**Model seems to forget earlier instructions**
- Likely hitting folds/prunes. Reduce context size (fewer files, shorter templates).
- Check `logs/baseline_gate.log` for fold frequency.
- Consider raising the safety factor in `capacity_calculator` (e.g., `0.85` → `0.90`).

**Tool outputs get truncated**
- Per-file token cost may be too high. Use TEMPLATE-style summaries.
- Re-measure `per_file_tokens` and update anchors in `scripts/capacity_calculator.py`.

**Semantic router feels slow or expensive**
- It's expected: embeddings add token and compute cost.
- Use only in `full` profile, and only when you need semantic search.
- Keep `lite` as your default for 1.5B work.

## Extending the harness

### Adding new tools/templates
1. Re-measure `guidance`/`per_file`/`delta` after adding the tool.
2. Update `ANCHORS` in `scripts/capacity_calculator.py`.
3. Run `python scripts/capacity_calculator.py --model 1.5b --verify` to confirm predictions.
4. Run the full baseline gate to ensure no regressions.

### Adding new models
1. Add a new entry to `MODEL_PRESETS` in `core/context.py:28–34`.
2. Define `max_tokens`, `safety`, `pruner_budget`, `max_messages`, `rounds_per_file`, `max_tool_result_bytes`.
3. Run `python scripts/capacity_calculator.py --model <new> --verify` with live benchmarks.
4. Update the baseline gate thresholds if needed.

## Summary

- Default to `lite` for 1.5B work; it's token-efficient and deterministic.
- Use `full` sparingly for debugging, tracing, or semantic search.
- Monitor capacity with `python scripts/capacity_calculator.py --model 1.5b --quiet`; expect **19** files for 1.5B.
- Watch for folds; reduce context size if the model loses track of instructions.
- Run the baseline gate before merging changes that affect plugins, calibration, or routing.

This harness is designed to keep 1.5B models productive without token blow-up or silent regressions. Stick to the invariants, and you can iterate safely.

## Related docs

- `docs/baseline_contract.md` — baseline contract and verification commands
- `docs/baseline_validation.md` — proof artifacts and invariant details
- `AGENTS.md` — repository-wide invariants and development workflow
- `PROJECT_TRACKING.md` — project state and open items
