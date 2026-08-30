# PROJECT_TRACKING — Cordi v2

## Purpose
Lite-first local agent for 1.5B models (qwen2.5-coder:1.5b). Pool philosophy: deterministic SymPy/datetime/units handlers run zero-token; only hard problems hit Ollama. Fix corditelite's violations: false zero-token (semantic router), double event emit, dual pruner destroying assistant reasoning, prompt injection as system, and 43-plugin complexity for a 4k ctx model.

## File Inventory

### Core (8 files) — complete
| File | Purpose | Status |
|---|---|---|
| `core/context.py` | Context + EventBus | complete, no change |
| `core/registry.py` | Plugin topology sort | complete |
| `core/summarizer.py` | **FIXED** fold_messages preserves assistant tool_calls | complete |
| `core/context_pruner.py` | **FIXED** 3000 budget, single authoritative pruner | complete |
| `core/event_log.py` | **FIXED** base64+zlib snapshot | complete |
| `core/messages.py` | Message dataclass | complete |
| `core/errors.py` | Error types | complete |
| `core/failure_taxonomy.py` | Failure classification | complete |

### Plugins/Agent (6) — complete
| File | Status |
|---|---|
| `plugins/agent/loop.py` | **FIXED** gated semantic, 3k budget, single pruner, turn.round, hardened injections |
| `plugins/agent/query_splitter.py` | **FIXED** strong/weak signals collision-free |
| `plugins/agent/semantic_router.py` | **FIXED** default OFF, config gated, cache entries nesting |
| `plugins/agent/parameter_extractor.py` | complete |
| `plugins/agent/multi_domain_router.py` | complete |
| `plugins/agent/aggregate_response.py` | complete |

### Plugins/Math (6) — complete
MathRouter/Pipeline/SymbolicEngine/Verifier, DateTimeRouter/Engine, UnitsRouter/Engine — deterministic, verified via sympy

### Plugins/Model (2) — fixed
| `plugins/model/embedding.py` | **FIXED** entries nesting | complete |
| `plugins/model/ollama.py` | complete |

### Plugins/Core 20 — gated to full profile
ContextPrunerPlugin always on; other 19 only in `--profile full` (main.py). No stubs.

### Root
| `main.py` | **FIXED** lite/full profiles | complete |
| `requirements.txt` | **FIXED** added sympy/gradio | complete |
| `.gitignore` | **FIXED** .cache/*.db | complete |
| `README.md` | updated | complete |

## Open Dependencies
- Ollama running at 127.0.0.1:11434 for live model tests (optional)
- sympy for math engine (now in requirements)
- gradio for ui.py

## Known Gaps
- None — no TODO/NotImplemented. All files runnable.
- SemanticRouter embeddings still cost tokens when enabled; lite keeps it OFF.
- Embedding cache still file-based (plaintext) but gitignored; not encrypted.

## Verification
- `pytest tests/test_math.py` — 42 tests, should pass (pool no change)
- `pytest tests/test_agent.py` with basetemp outside OneDrive
- Manual: `python main.py --profile lite` vs `full`
