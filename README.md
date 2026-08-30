# Cordi v2 — Lite-First Local Agent

Fork of corditelite with first-principles fixes for 1.5B local models (qwen2.5-coder:1.5b, 4k ctx).

## What changed vs corditelite (audit fixes)

**P0 — Zero-token guarantee**
- `SemanticRouter` default OFF. Enable only with `--enable-semantic-router` or `config["semantic_router_enabled"]=True`. No embedding cost by default.
- Fix double `turn.start` emit in `plugins/agent/loop.py:239 + 319` → single outer + `turn.round` per iteration.
- `requirements.txt` now complete (`sympy, gradio, ollama`).

**P1 — Context + security**
- Single pruner: `ContextPruner` @ 3000 token budget (was dual 4000 + 4000). Preserves `assistant` with `tool_calls` for 1.5B coherence (`core/summarizer.py:64`).
- Prompt injections hardened: injected as `user` with `[injected context]` prefix, not `system` (`plugins/agent/loop.py:326`).
- `.gitignore` + `EventLog` snapshot now `base64+zlib` (was latin-1 fragile), cache `entries` nesting avoids `"version"` key collision.

**P2 — Complexity**
- `main.py` `--profile lite|full` (default `lite` = 16 plugins vs 31). Lite is the real cordis-lite.
- `QuerySplitter` hardened: strong vs weak signals, collision-free (`add 5 days` → datetime, `add 2+3` → math, `time` alone → general).

## Architecture — Lite (default)
```
AgentLoop → Pool (zero-token) → Barista (Ollama)
Pool: ParameterExtractor → QuerySplitter → MultiDomainRouter → AggregateResponse
      MathRouter/Pipeline/Engine/Verifier, DateTimeRouter/Engine, UnitsRouter/Engine
Single ContextPruner (3000 budget)
```
Full adds: EmbeddingModel, SemanticRouter (gated), Health/Tracing/Metrics, Persona/Lifecycle/Telemetry, Memory, etc.

## Quick Start
```bash
pip install -r requirements.txt
python main.py --profile lite              # 16 plugins, default
python main.py --profile full --enable-semantic-router  # 31 plugins + embeddings
# or via python chat.py / ui.py
```

Commands: `/math`, `/datetime`, `/units`, `/help`, `/quit`

Multi-domain: `What's the derivative of x squared? And how is pizza made?` → `[math] 2*x` + `[general] ...`

## Testing
```bash
pytest tests/ -q --basetemp C:\tmp\pytest
# or
python -m pytest tests/test_math.py -q
```

## Profiles
| Profile | Plugins | Use |
|---|---|---|
| lite (default) | 16 | Local 1.5B, 4k ctx, save tokens |
| full | 31 | Debug, memory, observability |

See `PROJECT_TRACKING.md` for file inventory.
