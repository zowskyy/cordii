from __future__ import annotations

import sys
from pathlib import Path

from core.context import Context
from core.registry import PluginRegistry
from core.messages import Message
from plugins.agent.routers import try_datetime_router, try_math_router, try_units_router
from plugins.math.symbolic_engine import SymbolicEnginePlugin
from plugins.math.verifier import MathVerifierPlugin
from plugins.math.pipeline import MathPipelinePlugin
from plugins.math.datetime_router import DateTimeRouterPlugin
from plugins.math.datetime_engine import DateTimeEnginePlugin
from plugins.math.units_router import UnitsRouterPlugin
from plugins.math.units_engine import UnitsEnginePlugin
from plugins.model.ollama import OllamaModel


def build_ctx() -> Context:
    workspace = Path("workspace")
    ctx = Context(config={
        "workspace": str(workspace.resolve()),
        "model": "qwen2.5-coder:1.5b",
        "ollama_url": "http://127.0.0.1:11434",
    })
    reg = PluginRegistry(ctx)
    plugin_config = {
        "ollama_model": {"model": "qwen2.5-coder:1.5b", "base_url": "http://127.0.0.1:11434"},
    }
    reg.register_class(MathRouterPlugin)
    reg.register_class(SymbolicEnginePlugin)
    reg.register_class(MathVerifierPlugin)
    reg.register_class(MathPipelinePlugin)
    reg.register_class(DateTimeRouterPlugin)
    reg.register_class(DateTimeEnginePlugin)
    reg.register_class(UnitsRouterPlugin)
    reg.register_class(UnitsEnginePlugin)
    reg.register_class(OllamaModel, **plugin_config["ollama_model"])
    reg.start_all()
    return ctx


def try_router(text: str, ctx: Context) -> str | None:
    for router in (try_math_router, try_datetime_router, try_units_router):
        result = router(text, ctx)
        if result is not None:
            return result
    return None


def main() -> int:
    print("Loading plugins and model...")
    ctx = build_ctx()
    model: OllamaModel = ctx.plugins["ollama_model"]
    print(f"Model: {model.model}")
    print("Commands: /math <expr> /datetime <expr> /units <expr> /quit")
    print()

    history: list[Message] = []

    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not text:
            continue
        if text == "/quit":
            break
        if text == "/help":
            print("/math <expr>   e.g. /math solve x**2-4=0")
            print("/datetime <expr>  e.g. /datetime today")
            print("/units <expr>  e.g. /units convert 100 km to miles")
            print("/quit          exit")
            continue

        # 1. Try deterministic routers first
        router_out = try_router(text, ctx)
        if router_out is not None:
            print(router_out)
            continue

        # 2. Fallback to LLM chat
        history.append(Message(role="user", content=text))
        try:
            response = model.chat(history, [])
            history.append(Message(role="assistant", content=response.content))
            print(response.content)
        except Exception as exc:
            print(f"Error: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
