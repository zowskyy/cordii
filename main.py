from __future__ import annotations

import argparse
from pathlib import Path

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


def build_application(workspace: Path, model_name: str, ollama_url: str, db_path: Path, profile: str = "lite", enable_semantic_router: bool = False):
    """
    P2 FIX: Lite vs Full profile.
    - lite (default): only what saves tokens for 1.5B. 16 plugins vs 31.
      Zero-token pool + file tools + loop + single pruner. No observability theater.
    - full: adds all 31 plugins for debugging/analysis. Semantic router gated even here.
    """
    # Pass semantic flag via context config so SemanticRouter and loop can gate
    ctx = Context(config={
        "workspace": str(workspace.resolve()),
        "model": model_name,
        "ollama_url": ollama_url,
        "profile": profile,
        "semantic_router_enabled": enable_semantic_router,
    })
    reg = PluginRegistry(ctx)

    plugin_config = {
        "event_logger": {"db_path": db_path},
        "ollama_model": {"model": model_name, "base_url": ollama_url},
        "file_tools": {"workspace": workspace},
    }

    # === LITE CORE: always on ===
    reg.register(EventLogger(db_path))
    reg.register(OllamaModel(**plugin_config["ollama_model"]))
    reg.register(FileTools(workspace))
    reg.register(OllamaToolCallParser())
    # Deterministic pool (zero-token)
    reg.register(MathRouterPlugin())
    reg.register(SymbolicEnginePlugin())
    reg.register(MathVerifierPlugin())
    reg.register(MathPipelinePlugin())
    reg.register(DateTimeRouterPlugin())
    reg.register(DateTimeEnginePlugin())
    reg.register(UnitsRouterPlugin())
    reg.register(UnitsEnginePlugin())
    # Routing layer
    reg.register(ParameterExtractor())
    reg.register(QuerySplitter())
    reg.register(MultiDomainRouter())
    reg.register(AggregateResponse())
    # Single pruner (3k budget for 4k ctx) — needed even in lite
    reg.register(ContextPrunerPlugin())
    # Core loop
    reg.register(AgentLoop())
    reg.register(TerminalUI())

    if profile == "full":
        # Observability / extended memory — only in full
        reg.register(EmbeddingModel())
        # Semantic router gated: only enabled if flag set
        reg.register(SemanticRouter(enabled=enable_semantic_router))
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
        reg.register(ConfigValidationPlugin())
        reg.register(CIPlugin())
    else:
        # lite: still provide minimal health/metrics via loop internals, but don't register heavy plugins
        # Semantic router not registered in lite at all — zero embedding cost
        pass

    reg.start_all()
    return ctx, reg


def main() -> int:
    p = argparse.ArgumentParser(description="Cordis-Lite local coding agent (cordiiv2)")
    p.add_argument("--workspace", default="workspace", help="Workspace directory")
    p.add_argument("--model", default="qwen2.5-coder:1.5b", help="Ollama model name")
    p.add_argument("--ollama-url", default="http://127.0.0.1:11434", help="Ollama base URL")
    p.add_argument("--db", default="continuity/continuity.db", help="SQLite event log path")
    p.add_argument("--profile", default="lite", choices=["lite", "full"], help="lite=16 plugins (default, saves tokens), full=31 plugins (debug)")
    p.add_argument("--enable-semantic-router", action="store_true", help="Enable semantic router (embedding cost, off by default)")
    args = p.parse_args()
    workspace = Path(args.workspace).expanduser().resolve()
    db_path = Path(args.db).expanduser().resolve()
    ctx, reg = build_application(workspace, args.model, args.ollama_url, db_path, profile=args.profile, enable_semantic_router=args.enable_semantic_router)
    try:
        ctx.plugins["terminal_ui"].run()
    finally:
        reg.stop_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
