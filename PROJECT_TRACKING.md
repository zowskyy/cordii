# PROJECT_TRACKING — Cordi v2

## Purpose
Lite-first local agent for 1.5B models (qwen2.5-coder:1.5b). Pool philosophy: deterministic SymPy/datetime/units handlers run zero-token; only hard problems hit Ollama. Fix corditelite's violations: false zero-token (semantic router), double event emit, dual pruner destroying assistant reasoning, prompt injection as system, and 43-plugin complexity for a 4k ctx model.

## Current State
- **Audit findings: all P0/P1 resolved** — hardcoded API key (removed, env-based), ungated LLM routing in lite (gated + tested), parser name mismatch (fixed), ContextPrunerPlugin 4000 default (calibration-aligned), linter dot-path blind spot (fixed + regression test), broken `chat.py` (deleted).
- **Calibration separation** — model-specific numbers live only in `core/context.py` `MODEL_PRESETS`; invariant layers read via `calibration_from_context()` (AGENTS.md axiom: "Calibration separation").
- **De-Kilo** — `.kilo/`, `kilo.json`, and the AGENTS.md Kilo section removed; methodology lives in AGENTS.md.
- **Zero-token guarantee** — enforced at both LLM routing sites (semantic router, multi-domain fallback); gate test: `tests/test_agent.py::test_multi_domain_llm_fallback_gated_by_profile_and_flag`.
- **Core hardening (post-baseline)** — single tool results truncate to the per-model calibrated cap `max_tool_result_bytes` (window protection: one result can never swallow the 4k window); `_call_llm_directly` fails loud instead of appending a silently empty fragment; the pruner is now dual-metric (count pass + token pass that enforces `token_budget` even under the message cap, protecting leading system + last two, dropping lowest-score first). Tests: `test_tool_result_truncated_to_calibrated_limit`, `test_tool_result_limit_follows_calibration_override`, `test_pruner_token_pass_when_under_message_limit`.

## File Inventory (verified counts)
| Tree | Py files | Notes |
|---|---|---|
| `core/` | 30 | plugin base, registry, context (incl. calibration table), pruner, summarizer, event log, reality projector, linting, … |
| `plugins/agent/` | 9 | `loop.py` + `routers.py` (deterministic dispatch), `specialized_routers.py`, `query_splitter`, `semantic_router` (gated), `multi_domain_router`, `parameter_extractor`, `aggregate_response` |
| `plugins/core/` | 23 | 22 plugins + `__init__`; registered in `main.py` lite/full blocks |
| `plugins/math/` | 9 | router/pipeline/symbolic_engine/verifier + datetime + units (deterministic) |
| `plugins/model/` | 4 | `ollama` (lite), `embedding` (full), `ensemble` (benchmark-only), `__init__` |
| `plugins/tools/` | 2 | `FileTools` |
| `benchmark/` | 9 | R&D/eval track — ratified exception (AGENTS.md "Ratified auxiliary trees") |
| `tests/` | 40 + 9 | top-level + `failure_recovery/` package (harness, live/fault tests, debug runner) |
| `scripts/` | 2 | `capacity_calculator.py`, `capacity_diff.py` |
| root | 2 py | `main.py`, `ui.py` (ratified Gradio harness); `run.ps1` / `run-qwen.ps1` launchers |

## Milestone: v2.0-baseline-stable (2026-08-30)
- **Model:** `qwen2.5-coder:1.5b`
- **Deterministic suite:** 279 passed, 7 skipped
- **Live integration:** 4/4 passed in ~9.47s
- **Validated invariants:**
  - 1.5B baseline stable
  - 4k context hygiene (pruner token+message pass, tool-result truncation)
  - Zero-token `lite` profile
  - Calibration centralization + validation (`validate_calibration` wired into `resolve_calibration`)
  - Profile isolation (`lite` excludes `semantic_router`/`embedding_model`)
  - Injection hardening (`[injected context]` prefix, `user` role)
  - Event hygiene (exactly one `turn.start`/`turn.end`, one `turn.round` per iteration)

## Open Items (honest)
- `requirements.txt` pins `pytest<9`; the suite is also verified on 9.1.1 (works, unpinned).
- The literal verification gate (`--basetemp C:\tmp\pytest_cordiiv2`) needs a plain shell: under the DSH workspace-write sandbox `C:\tmp` is file-write-readonly and `os.mkdir(0o700)` gets hostile ACLs, breaking `tempfile.mkdtemp`. Run with a writable `TEMP` instead — same suite, same result.
- Live 1.5B tests are probabilistic (axiom: the model is flaky). Flakiness guards match the exact max-rounds message (`plugins/agent/loop.py:568`); the deterministic suite is the hard gate, live runs are confirmatory.
- Embedding cache is file-based plaintext (gitignored), not encrypted.
- SemanticRouter embeddings cost tokens when enabled (full + `--enable-semantic-router` only); lite keeps it OFF.

## Verification
- **Hard gate (deterministic):** `pytest --basetemp C:\tmp\pytest_cordiiv2` → 279 passed, 7 skipped
- **Live (confirmatory):** `pytest --basetemp C:\tmp\pytest_cordiiv2 --live` → 280 passed, 3 skipped (4/4 live 1.5B integration tests green)
- **gitignore gate:** `git check-ignore` confirms `.cache/*`, `continuity/*.db`, `*.db`, `workspace/*` ignored (`.gitignore`).
- **Baseline gate:** `powershell -File scripts/baseline_gate.ps1` (re-runs deterministic + live gates with assertions).
- **Manual:** `python main.py --profile lite` vs `full`; `python scripts/capacity_calculator.py --model 1.5b`.
