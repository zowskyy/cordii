# Baseline Validation — Cordi v2

## Milestone
- **Tag:** `v2.0-baseline-stable`
- **Commit:** `f9cef7b`
- **Date:** 2026-08-30
- **Model:** `qwen2.5-coder:1.5b`

## Proof Artifacts

### Deterministic suite
```powershell
pytest --basetemp C:\tmp\pytest_cordiiv2 -q
```
**Expected:** 276 passed, 7 skipped

### Live integration suite
```powershell
pytest --basetemp C:\tmp\pytest_cordiiv2 --live -k integration -q
```
**Expected:** 4 passed (all live 1.5B integration tests green)

### Baseline gate script
```powershell
powershell -File scripts/baseline_gate.ps1
```
Runs both suites and asserts the counts above. Add `-SkipLive` to run only the deterministic gate.

## Invariants Proven
- **1.5B stability** — single-pruner token+message pass, tool-result truncation to calibrated byte cap
- **4k context hygiene** — pruner budget 3000, KV headroom 1000, folds trigger on tokens OR messages
- **Zero-token `lite`** — 19 plugins, deterministic routing only, no LLM calls unless `full` + `--enable-semantic-router`
- **Calibration centralization** — `validate_calibration()` wired into `resolve_calibration()`; all per-model numbers sourced from `core/context.py` `MODEL_PRESETS`
- **Profile isolation** — `lite` excludes `semantic_router` and `embedding_model`; `AgentLoop._semantic_router` is `None`
- **Injection hardening** — injections inserted as `user` with `[injected context]` prefix
- **Event hygiene** — exactly one `turn.start`/`turn.end`, one `turn.round` per iteration

## Regression Guardrails
- Any change that drops deterministic pass count below 276 is a baseline regression.
- Any change that drops live pass count below 4 is a baseline regression.
- The deterministic suite is the hard gate; live runs are confirmatory (1.5B is flaky by nature).
