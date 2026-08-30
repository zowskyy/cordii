"""
Hydrate CLI — programmatic interface to the Cordis-Lite pool.

Registers all plugins and exposes them as commands.
Lets Kilo delegate work to plugins instead of inline bash/edit.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from core.context import Context
from core.registry import PluginRegistry
from plugins.agent.loop import AgentLoop
from plugins.agent.semantic_router import SemanticRouter
from plugins.agent.parameter_extractor import ParameterExtractor
from plugins.agent.query_splitter import QuerySplitter
from plugins.agent.multi_domain_router import MultiDomainRouter
from plugins.agent.aggregate_response import AggregateResponse
from plugins.core.event_logger import EventLogger
from plugins.core.persona_memory import PersonaMemoryPlugin
from plugins.core.lifecycle import LifecycleConsolidatorPlugin
from plugins.core.telemetry import TelemetryPlugin
from plugins.core.context_builder import ContextBuilderPlugin
from plugins.core.context_pruner import ContextPrunerPlugin
from plugins.core.error_recovery import ErrorRecoveryPlugin
from plugins.core.health_monitoring import HealthMonitoringPlugin
from plugins.core.metrics import MetricsPlugin
from plugins.core.tracing import TracingPlugin
from plugins.core.formal_contracts import FormalContractsPlugin
from plugins.core.hot_swap import HotSwapPlugin
from plugins.core.semantic_memory import SemanticMemoryPlugin
from plugins.core.episodic_memory import EpisodicMemoryPlugin
from plugins.core.closed_loop import ClosedLoopRetrievalPlugin
from plugins.core.reality_projector import RealityProjectorPlugin
from plugins.core.recovery_manager import RecoveryManagerPlugin
from plugins.core.config_validation import ConfigValidationPlugin
from plugins.core.linting import LintingPlugin
from plugins.core.logic_layer import LogicLayerPlugin
from plugins.core.intent_router import IntentRouterPlugin
from plugins.core.summarizer import SummarizerPlugin
from plugins.ci.ci_plugin import CIPlugin
from plugins.math.router import MathRouterPlugin
from plugins.math.symbolic_engine import SymbolicEnginePlugin
from plugins.math.verifier import MathVerifierPlugin
from plugins.math.pipeline import MathPipelinePlugin
from plugins.math.datetime_router import DateTimeRouterPlugin
from plugins.math.datetime_engine import DateTimeEnginePlugin
from plugins.math.units_router import UnitsRouterPlugin
from plugins.math.units_engine import UnitsEnginePlugin
from plugins.model.ollama import OllamaModel
from plugins.model.embedding import EmbeddingModel
from plugins.parsers.tool_call_parser import OllamaToolCallParser
from plugins.tools.file import FileTools
from plugins.ui.terminal import TerminalUI


def build_pool(workspace: Path, model_name: str, ollama_url: str, db_path: Path) -> tuple[Context, PluginRegistry]:
    ctx = Context(config={"workspace": str(workspace.resolve()), "model": model_name, "ollama_url": ollama_url})
    reg = PluginRegistry(ctx)

    plugin_config = {
        "event_logger": {"db_path": db_path},
        "ollama_model": {"model": model_name, "base_url": ollama_url},
        "file_tools": {"workspace": workspace},
    }

    reg.register(EventLogger(db_path))
    reg.register(OllamaModel(**plugin_config["ollama_model"]))
    reg.register(EmbeddingModel())
    reg.register(HealthMonitoringPlugin())
    reg.register(MetricsPlugin())
    reg.register(LintingPlugin())
    reg.register(IntentRouterPlugin())
    reg.register(LogicLayerPlugin())
    reg.register(TracingPlugin())
    reg.register(PersonaMemoryPlugin())
    reg.register(LifecycleConsolidatorPlugin())
    reg.register(TelemetryPlugin())
    reg.register(SemanticMemoryPlugin())
    reg.register(EpisodicMemoryPlugin())
    reg.register(RealityProjectorPlugin())
    reg.register(RecoveryManagerPlugin())
    reg.register(SummarizerPlugin())
    reg.register(ErrorRecoveryPlugin())
    reg.register(FormalContractsPlugin())
    reg.register(HotSwapPlugin())
    reg.register(ClosedLoopRetrievalPlugin())
    reg.register(ContextBuilderPlugin())
    reg.register(ContextPrunerPlugin())
    reg.register(ConfigValidationPlugin())
    reg.register(SemanticRouter())
    reg.register(ParameterExtractor())
    reg.register(QuerySplitter())
    reg.register(MultiDomainRouter())
    reg.register(AggregateResponse())
    reg.register(CIPlugin())
    reg.register(OllamaToolCallParser())
    reg.register(FileTools(workspace))
    reg.register(AgentLoop())
    reg.register(MathRouterPlugin())
    reg.register(SymbolicEnginePlugin())
    reg.register(MathVerifierPlugin())
    reg.register(MathPipelinePlugin())
    reg.register(DateTimeRouterPlugin())
    reg.register(DateTimeEnginePlugin())
    reg.register(UnitsRouterPlugin())
    reg.register(UnitsEnginePlugin())
    reg.register(TerminalUI())

    reg.start_all()
    return ctx, reg


def cmd_status(ctx: Context, _args: Any) -> None:
    plugins = ctx.plugins
    print(f"Pool: {len(plugins)} plugins registered")
    for name, plugin in sorted(plugins.items()):
        deps = list(getattr(plugin, "dependencies", ()))
        print(f"  {name} deps={deps}")


def cmd_ci(ctx: Context, _args: Any) -> None:
    ci = ctx.plugins.get("ci_plugin")
    if ci is None:
        print("CI plugin not available")
        return
    status = ci.get_status()
    print(f"CI: {status['message']}")
    if status.get("url"):
        print(f"  {status['url']}")


def cmd_event(ctx: Context, args: Any) -> None:
    event_type = args.event_type
    payload = json.loads(args.payload) if args.payload else {}
    ctx.events.emit(event_type, payload)
    print(f"Emitted: {event_type} {payload}")


def cmd_math(ctx: Context, args: Any) -> None:
    pipeline = ctx.plugins.get("math_pipeline")
    if pipeline is None:
        print("math_pipeline not available")
        return
    result = pipeline.run(args.expr)
    if result.success:
        print(f"Result: {result.result}")
        for step in result.steps:
            print(f"  {step}")
    else:
        print(f"Error: {result.error}")


def cmd_agent(ctx: Context, args: Any) -> None:
    agent = ctx.plugins.get("agent_loop")
    if agent is None:
        print("agent_loop not available")
        return
    answer = agent.run(args.query)
    print(answer)


def main() -> int:
    p = argparse.ArgumentParser(description="Hydrate CLI — delegate to the pool")
    p.add_argument("--workspace", default="workspace")
    p.add_argument("--model", default="qwen2.5-coder:1.5b")
    p.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    p.add_argument("--db", default="continuity/continuity.db")
    sub = p.add_subparsers(dest="command")

    sub.add_parser("status")
    sub.add_parser("ci")

    ev = sub.add_parser("event")
    ev.add_argument("event_type")
    ev.add_argument("payload", nargs="?")

    math = sub.add_parser("math")
    math.add_argument("expr")

    agent = sub.add_parser("agent")
    agent.add_argument("query")

    args = p.parse_args()
    if not args.command:
        p.print_help()
        return 0

    workspace = Path(args.workspace).expanduser().resolve()
    db_path = Path(args.db).expanduser().resolve()
    ctx, reg = build_pool(workspace, args.model, args.ollama_url, db_path)

    try:
        if args.command == "status":
            cmd_status(ctx, args)
        elif args.command == "ci":
            cmd_ci(ctx, args)
        elif args.command == "event":
            cmd_event(ctx, args)
        elif args.command == "math":
            cmd_math(ctx, args)
        elif args.command == "agent":
            cmd_agent(ctx, args)
    finally:
        reg.stop_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
