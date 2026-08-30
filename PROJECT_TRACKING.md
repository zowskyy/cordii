# PROJECT_TRACKING — Cordi v2

## Purpose
Lite-first local agent for 1.5B models (qwen2.5-coder:1.5b). Pool philosophy: deterministic SymPy/datetime/units handlers run zero-token; only hard problems hit Ollama. Fix corditelite's violations: false zero-token (semantic router), double event emit, dual pruner destroying assistant reasoning, prompt injection as system, and 43-plugin complexity for a 4k ctx model.

## Current State
- **Audit findings: all P0/P1 resolved** — hardcoded API key (removed, env-based), ungated LLM routing in lite (gated + tested), parser name mismatch (fixed), ContextPrunerPlugin 4000 default (calibration-aligned), linter dot-path blind spot (fixed + regression test), broken `chat.py` (deleted).
- **Calibration separation** — model-specific numbers live only in `core/context.py` `MODEL_PRESETS`; invariant layers read via `calibration_from_context()` (AGENTS.md axiom: "Calibration separation").
- **De-Kilo** — `.kilo/`, `kilo.json`, and the AGENTS.md Kilo section removed; methodology lives in AGENTS.md.
- **Zero-token guarantee** — enforced at both LLM routing sites (semantic router, multi-domain fallback); gate test: `tests/test_agent.py::test_multi_domain_llm_fallback_gated_by_profile_and_flag`.

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

## Open Items (honest)
- `requirements.txt` pins `pytest<9`; the suite is also verified on 9.1.1 (works, unpinned).
- The literal verification gate (`--basetemp C:\tmp\pytest_cordiiv2`) needs a plain shell: under the DSH workspace-write sandbox `C:\tmp` is file-write-readonly and `os.mkdir(0o700)` gets hostile ACLs, breaking `tempfile.mkdtemp`. Run with a writable `TEMP` instead — same suite, same result.
- Live 1.5B tests are probabilistic (axiom: the model is flaky). Flakiness guards match the exact max-rounds message (`plugins/agent/loop.py:568`); the deterministic suite is the hard gate, live runs are confirmatory.
- Embedding cache is file-based plaintext (gitignored), not encrypted.
- SemanticRouter embeddings cost tokens when enabled (full + `--enable-semantic-router` only); lite keeps it OFF.

## Verification
- **Hard gate (deterministic):** `pytest` → 256 passed, 7 skipped (the 7 live tests skip without `--live`).
- **Live (confirmatory):** `pytest --live` with Ollama at 127.0.0.1:11434 → 263 passed (all 7 live 1.5B tests green in the latest run).
- **gitignore gate:** `git check-ignore` confirms `.cache/*`, `continuity/*.db`, `*.db`, `workspace/*` ignored (`.gitignore`).
- **Manual:** `python main.py --profile lite` vs `full`; `python scripts/capacity_calculator.py --model 1.5b`.
