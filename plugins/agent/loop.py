from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

from core.context import Context
from core.calibration import MODEL_PRESETS, DEFAULT_PRESET_KEY, calibration_from_context
from core.context_pruner import ContextPruner, PrunedContext
from core.errors import ToolError
from core.events import Manifest, SYSTEM_MESSAGE, USER_MESSAGE, TOOL_CALL_START, TOOL_CALL_END
from core.failure_taxonomy import FailureClassifier, FailureType, PreFlightGuard
from core.messages import Message
from core.plugin import Plugin
from core.retry import RetryPolicy, retry_with_backoff
from core.reality import RealityProjector, RequestEnvelope
from core.self_healing import BudgetedSelfHealing
from core.summarizer import Summarizer
from plugins.agent.routers import try_datetime_router, try_math_router, try_units_router
from plugins.agent.semantic_router import SemanticRouter
from plugins.agent.specialized_routers import RepairMessageBuilder, SpecializedRouters, ToolResultVerifier
from plugins.agent.aggregate_response import AggregateResponse
from plugins.agent.multi_domain_router import MultiDomainRouter
from plugins.agent.schema_router import SchemaRouter
from plugins.core.decision_logger import DecisionLogger

# Back-compat alias (tests import it). The VALUE comes from the shared model
# calibration table (core.context.MODEL_PRESETS), not a literal here: the
# invariant layer carries no model-specific numbers. 1.5b default = 30000
# budget leaving ~2768 headroom for the 32768 window (Modelfile num_ctx 32768).
TOKEN_BUDGET = MODEL_PRESETS[DEFAULT_PRESET_KEY]["pruner_budget"]


class AgentLoop(Plugin):
    name = "agent_loop"
    dependencies = ("ollama_model", "file_tools")

    def __init__(self, max_rounds: int = 12, stream: bool = False) -> None:
        super().__init__()
        self.max_rounds = max_rounds
        self.stream = stream
        self._tool_handlers: dict[str, Callable[[dict[str, Any]], str]] = {}
        self._tool_schemas: list[dict[str, Any]] = []
        self._failed_calls: dict[str, int] = {}
        self._successful_calls: set[str] = set()
        self._replan_count = 0
        self._round = 0
        self._healing = BudgetedSelfHealing()
        self._context_builder = None
        self._retry_policy = RetryPolicy()
        # Built in start() from the active model calibration (core.context).
        self._context_pruner: ContextPruner | None = None
        self._token_budget = TOKEN_BUDGET
        self._max_result_bytes = MODEL_PRESETS[DEFAULT_PRESET_KEY]["max_tool_result_bytes"]
        self._parse_retry_count = 0
        self._routers: SpecializedRouters | None = None
        self._semantic_router: SemanticRouter | None = None
        self._multi_domain_router: MultiDomainRouter | None = None
        self._aggregator: AggregateResponse | None = None
        self._multi_domain_results: list[Any] = []
        self._projector: RealityProjector | None = None
        self._manifest: Manifest | None = None
        self._system_prompt: str | None = None
        self._schema_router: SchemaRouter | None = None
        self._array_helper: Any = None
        self._app_verifier: Any = None
        self._decision_logger: DecisionLogger | None = None

    def start(self) -> None:
        assert self.context is not None
        cal = calibration_from_context(self.context)
        self._token_budget = cal["pruner_budget"]
        self._max_result_bytes = cal["max_tool_result_bytes"]
        self._context_pruner = ContextPruner(max_messages=cal["max_messages"], token_budget=cal["pruner_budget"])
        files = self.context.plugins["file_tools"]
        self._tool_schemas = files.schemas()
        self._tool_handlers = {s["function"]["name"]: files.execute for s in self._tool_schemas}
        self._schema_router = self.context.plugins.get("schema_router")
        self._array_helper = self.context.plugins.get("array_helper")
        self._routers = SpecializedRouters(
            tool_handlers=self._tool_handlers,
            context=self.context,
            record_tool_result=self._record_tool_result,
            resolve_path=self._resolve_path,
        )
        self._semantic_router = self.context.plugins.get("semantic_router")
        self._multi_domain_router = self.context.plugins.get("multi_domain_router")
        self._aggregator = self.context.plugins.get("aggregate_response")
        self._app_verifier = self.context.plugins.get("app_verifier")
        self._decision_logger = self.context.plugins.get("decision_logger")
        if self._decision_logger is not None:
            self._decision_logger.set_session_id(self._get_session_id())
        self._tool_result_pruner = self.context.plugins.get("tool_result_pruner")

    def tool_schemas(self) -> list[dict[str, Any]]:
        return list(self._tool_schemas)

    def _get_active_tool_schemas(self) -> list[dict[str, Any]]:
        """Return the tool schemas the model should see this turn.

        If schema_router is enabled and compact_schema is True, use the
        compact schema from the router instead of the verbose registry schemas.
        """
        if self._schema_router is not None and getattr(self._schema_router, "enabled", False):
            compact = self._schema_router.get_model_tools()
            if compact:
                return compact
        return list(self._tool_schemas)

    def _expand_compact_calls(self, tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Expand compact-schema logical calls (call_tool) into real tool calls.

        When the model uses call_tool(tool, args), this method rewrites each
        call into the real tool name/arguments expected by the rest of the loop.
        Calls that are not compact-schema calls pass through unchanged.
        'done' logical calls are dropped (no real tool to execute).
        """
        if self._schema_router is None or not getattr(self._schema_router, "enabled", False):
            return tool_calls

        expanded: list[dict[str, Any]] = []
        for call in tool_calls:
            real = self._schema_router.expand_call(call)
            if real is None:
                # 'done' or unknown — skip (no real tool)
                continue
            expanded.append(real)
        return expanded

    def _sig(self, call: dict[str, Any]) -> str:
        fn = call.get("function", {})
        try:
            return f"{fn.get('name', '')}:{json.dumps(fn.get('arguments') or {}, sort_keys=True, default=str)}"
        except Exception:
            return f"{fn.get('name', '')}:{fn.get('arguments')}"

    def _blocked(self, sig: str) -> bool:
        return self._failed_calls.get(sig, 0) >= 2

    def _record_fail(self, sig: str) -> None:
        self._failed_calls[sig] = self._failed_calls.get(sig, 0) + 1

    def _log_routing_decision(self, decision_type: str, input_data: Any, output_data: Any) -> None:
        if self._decision_logger is not None:
            self._decision_logger.increment_turn()
            self._decision_logger.log_decision(
                decision_type=decision_type,
                input_data=input_data,
                output_data=output_data,
                confidence=1.0,
                duration_ms=0.0,
                alternatives_considered=[],
            )

    def _log_tool_decision(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: str,
        success: bool,
        duration_ms: float = 0.0,
        error: str | None = None,
    ) -> None:
        if self._decision_logger is not None:
            self._decision_logger.log_decision(
                decision_type="tool_selection",
                input_data={"tool_name": tool_name, "arguments": arguments},
                output_data={"result": result, "success": success, "error": error},
                confidence=1.0 if success else 0.0,
                duration_ms=duration_ms,
                alternatives_considered=[],
            )

    @staticmethod
    def _parse_arguments(raw: Any) -> dict[str, Any]:
        if raw is None:
            return {}
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ToolError(f"Malformed tool arguments: {raw!r}") from exc
            if not isinstance(parsed, dict):
                raise ToolError("Tool arguments must decode to a JSON object.")
            return parsed
        raise ToolError("Unsupported tool argument format.")

    def _classify_failure(self, exc: Exception, context: dict[str, Any]) -> FailureType:
        return FailureClassifier.classify(exc, context)

    def _quarantine(self, context: Any, sig: str, exc: Exception) -> None:
        note = json.dumps({
            "role": "system",
            "content": f"Previous attempt to call {sig} failed with {type(exc).__name__}: {exc}. Do not repeat identical call.",
        }, ensure_ascii=False)
        context.append_message("system", note)

    def _execute_tool_call(self, call: dict[str, Any]) -> str:
        fn = call.get("function", {})
        name = fn.get("name")
        if not isinstance(name, str):
            raise ToolError("Tool call has no valid function name.")

        # Defense-in-depth: if a compact-schema call_tool reaches dispatch
        # without prior expansion, expand it here.
        if name == "call_tool" and self._schema_router is not None and getattr(self._schema_router, "enabled", False):
            expanded = self._schema_router.expand_call(call)
            if expanded is None:
                # 'done' or unknown — no real tool to execute
                return json.dumps({"status": "done", "tool": "call_tool", "arguments": {}}, ensure_ascii=False)
            call = expanded
            fn = call.get("function", {})
            name = fn.get("name", "")

        arguments = self._parse_arguments(fn.get("arguments"))
        handler = self._tool_handlers.get(name)
        if handler is None:
            raise ToolError(f"Unknown tool requested by model: {name}")

        call_id = call.get("id") or f"call_{uuid.uuid4().hex[:8]}"
        signature = self._sig(call)

        if self._blocked(signature):
            error_msg = json.dumps({"error": f"Tool '{name}' blocked after repeated failures", "tool": name, "arguments": arguments, "blocked": True}, ensure_ascii=False)
            if self.context and self.context.plugins.get("event_logger"):
                self.context.plugins["event_logger"].emit("tool.result", {"tool_name": name, "call_id": call_id, "arguments": arguments, "success": False, "result": error_msg, "blocked": True})
            return error_msg

        logger = self.context.plugins.get("event_logger") if self.context else None
        step = None
        if logger is not None:
            step = logger.start_step(name, arguments)
            flags = PreFlightGuard.check(name, arguments, {"recent_tool_calls": []})
            if flags:
                step.governance_check_passed = False

        if self.context is not None:
            self.context.events.emit("tool.invoked", {"tool_name": name, "call_id": call_id, "arguments": arguments})
            self.context.events.emit(TOOL_CALL_START, {"tool_name": name, "call_id": call_id, "arguments": arguments})

        try:
            result = retry_with_backoff(
                handler,
                name,
                arguments,
                policy=self._retry_policy,
                should_retry=lambda exc: self._classify_failure(
                    exc, {"tool_name": name, "arguments": arguments}
                ) in (FailureType.TRANSIENT, FailureType.ARGUMENT),
            )
        except Exception as exc:
            if self.context is not None:
                self.context.events.emit(TOOL_CALL_END, {"tool_name": name, "call_id": call_id, "success": False, "error": str(exc)})
            if step is not None:
                logger.finish_step(step, error=str(exc))
            self._record_fail(signature)
            failure_type = self._classify_failure(exc, {"tool_name": name, "arguments": arguments})
            self._quarantine(self.context, signature, exc)
            healing = self._healing.handle_failure(failure_type, {"tool_name": name, "arguments": arguments})
            action = healing.get("action", "abstain")
            # retry_with_backoff already exhausted execution-layer retries;
            # preserve historical behavior by skipping the repair message on
            # retry-class failures.
            if action != "retry":
                repair = RepairMessageBuilder.build(name, failure_type, healing)
                self.context.append_message("system", repair)
            raise

        # Post-execute: prune or spill oversized results.
        pruned_result = result
        if self._tool_result_pruner is not None:
            try:
                pruned_result, _pruned, _spill = self._tool_result_pruner.prune(name, call_id, result)
            except Exception:
                pruned_result = result

        if step is not None:
            verified = ToolResultVerifier.verify(name, arguments, pruned_result)
            step.governance_check_passed = verified
            logger.finish_step(step, output=pruned_result)

        if self.context is not None:
            self.context.events.emit(TOOL_CALL_END, {"tool_name": name, "call_id": call_id, "success": True})

        return pruned_result

    def _get_session_id(self) -> str:
        event_logger = self.context.plugins.get("event_logger") if self.context else None
        if event_logger is not None and hasattr(event_logger, "continuity"):
            return event_logger.continuity.session_id
        return "default"

    def _build_manifest(self, system_prompt: str) -> Manifest:
        active_schemas = self._get_active_tool_schemas()
        sch_json = json.dumps(active_schemas, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        sch_hash = hashlib.sha256(sch_json.encode("utf-8")).hexdigest()
        profile = self.context.config.get("profile", "lite") if self.context else "lite"
        digest = hashlib.sha256(
            json.dumps(
                {
                    "system_prompt": system_prompt,
                    "tool_schema_hash": sch_hash,
                    "profile": profile,
                    "budget_tokens": self._token_budget,
                    "serializer_version": "v1",
                },
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return Manifest(
            digest=digest,
            tool_schema_hash=sch_hash,
            prompt_hash=hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
            serializer_version="v1",
            profile=profile,
            budget_tokens=self._token_budget,
        )

    def _compile_request_envelope(self, session_id: str, system_prompt: str) -> RequestEnvelope:
        if self._projector is None:
            self._projector = RealityProjector(self.context.plugins["event_logger"].event_log)
        if self._manifest is None:
            self._manifest = self._build_manifest(system_prompt)
        self._projector.invalidate_cache(session_id)
        active_schemas = self._get_active_tool_schemas()
        return self._projector.compile_request(session_id, self._manifest, system_prompt, active_schemas, self._token_budget)

    @staticmethod
    def _consume_stream(chunks: Any, on_stream: Optional[Callable[[Message], None]] = None) -> Message:
        final = None
        for chunk in chunks:
            final = chunk
            if on_stream is not None:
                on_stream(chunk)
        return final or Message("assistant", "")

    def _record_tool_result(self, tool_name: str, arguments: dict, result: str, success: bool) -> None:
        if isinstance(result, str):
            # 33k window protection (calibration-separation axiom): a single tool
            # result must never occupy more than its calibrated share of the
            # window. The file keeps its full content on disk; the model works
            # on the first chunk and the marker tells it the file continues.
            raw = result.encode("utf-8")
            if len(raw) > self._max_result_bytes:
                result = raw[: self._max_result_bytes].decode("utf-8", errors="ignore") + f"\n…[truncated: showing first {self._max_result_bytes} bytes of a larger file]"
            # Compact-schema result compression: truncate large read results
            if self._schema_router is not None and getattr(self._schema_router, "compact_mode", False):
                try:
                    parsed = json.loads(result)
                    if isinstance(parsed, dict):
                        result = json.dumps(self._schema_router.compress_result(parsed), ensure_ascii=False)
                    elif not isinstance(parsed, list) and len(result) > self._schema_router._PREVIEW_LIMIT:
                        result = result[: self._schema_router._PREVIEW_LIMIT] + "..."
                except (json.JSONDecodeError, TypeError):
                    if len(result) > self._schema_router._PREVIEW_LIMIT:
                        result = result[: self._schema_router._PREVIEW_LIMIT] + "..."
        call_id = f"zt_{tool_name}_{hashlib.md5(str(arguments).encode()).hexdigest()[:8]}"
        if self.context is not None:
            self.context.events.emit("tool.invoked", {"tool_name": tool_name, "call_id": call_id, "arguments": arguments})
            self.context.events.emit("tool.result", {"tool_name": tool_name, "call_id": call_id, "arguments": arguments, "success": success, "result": result})
        self.context.append_message("tool", result, tool_name=tool_name)

    def _resolve_path(self, user_path: str) -> Optional[Path]:
        file_tools = self.context.plugins.get("file_tools") if self.context else None
        if file_tools is None:
            return None
        try:
            return file_tools._resolve(user_path)
        except Exception:
            return None

    def _try_multi_domain(self, user_text: str) -> None:
        if self._multi_domain_router is None or self._aggregator is None:
            return

        multi = self._multi_domain_router.route_multi(user_text, self.context)
        if multi is None:
            return

        deterministic = [r for r in multi.results if r.response is not None]
        unresolved = [r for r in multi.results if r.response is None]

        if not deterministic:
            return

        # Zero-token guarantee (P0): the LLM fallback for unresolved fragments is a
        # routing LLM step — allowed ONLY in the full profile AND with explicit
        # --enable-semantic-router. Otherwise abandon multi-domain routing and let
        # the query fall through to the deterministic routers + normal agent loop.
        if unresolved and not (
            self.context.config.get("profile") == "full"
            and self.context.config.get("semantic_router_enabled", False)
        ):
            return

        if unresolved:
            for frag in unresolved:
                llm_answer = self._call_llm_directly(frag.fragment.text)
                deterministic.append(type(frag)(fragment=frag.fragment, domain=frag.domain, response=llm_answer))

        self._multi_domain_results = deterministic

    def _call_llm_directly(self, text: str) -> str:
        # Fail-loud: if the model call fails here (Ollama down, model error),
        # the main agent loop would fail too — surface the real error instead
        # of appending a silently empty fragment to the aggregated answer.
        model = self.context.plugins.get("ollama_model") if self.context else None
        if model is None:
            return ""
        response = model.chat([Message(role="user", content=text)], tools=[])
        return response.content or ""

    def run(self, user_text: str, on_stream: Optional[Callable[[Message], None]] = None) -> str:
        assert self.context is not None
        model = self.context.plugins["ollama_model"]
        summarizer = Summarizer()
        self._context_builder = self.context.plugins.get("context_builder")

        self.context.reset_cancel()
        self._failed_calls.clear()
        self._successful_calls.clear()
        self._replan_count = 0
        self._parse_retry_count = 0

        # Reset per-run state for optional plugins (zero-drag invariant)
        if self._array_helper is not None and hasattr(self._array_helper, "reset_run_state"):
            self._array_helper.reset_run_state()

        self.context.append_message("user", user_text)
        if self.context is not None:
            self.context.events.emit("user.message", {"content": user_text})

        session_id = self._get_session_id()
        self.context.events.emit("turn.start", {
            "user_text": user_text,
            "session_id": session_id,
        })

        self._try_multi_domain(user_text)

        if not self._multi_domain_results:
            fast_result = self._routers.try_zero_thought(user_text) if self._routers else None
            if fast_result is not None:
                self._log_routing_decision("zero_thought", user_text, fast_result)
                return fast_result

            # P0 FIX: SemanticRouter is gated (default OFF) to preserve zero-token guarantee
            # Only route semantically if enabled via config or constructor
            if self._semantic_router is not None and getattr(self._semantic_router, "enabled", False):
                semantic_result = self._semantic_router.route(user_text)
                if semantic_result is not None:
                    self._log_routing_decision("semantic_router", user_text, semantic_result)
                    return semantic_result

            math_result = try_math_router(user_text, self.context)
            if math_result is not None:
                self._log_routing_decision("math_router", user_text, math_result)
                return math_result

            datetime_result = try_datetime_router(user_text, self.context)
            if datetime_result is not None:
                self._log_routing_decision("datetime_router", user_text, datetime_result)
                return datetime_result

            units_result = try_units_router(user_text, self.context)
            if units_result is not None:
                self._log_routing_decision("units_router", user_text, units_result)
                return units_result

        if self._multi_domain_results:
            return self._aggregator.aggregate(self._multi_domain_results) if self._aggregator else str(self._multi_domain_results)

        task_state = {"goal": user_text, "files_touched": [], "tools_used": [], "unresolved_subtasks": []}

        # --- Optional ArrayHelper integration (zero-drag when irrelevant) ---
        # Analyze task relevance deterministically (reset already done in run() preamble)
        array_facts: dict[str, Any] = {}
        array_facts_changed = False

        if self._array_helper is not None:
            array_analysis = self._array_helper.analyze_task(user_text)
            if array_analysis.get("relevant") and array_analysis.get("confidence") in ("medium", "high"):
                array_facts = {
                    "operation": array_analysis.get("operation"),
                    "risks": array_analysis.get("risks", []),
                }
                array_facts_changed = True
                if self.context is not None:
                    self.context.events.emit("array.analysis.completed", {
                        "session_id": session_id,
                        "relevant": True,
                        "operation": array_analysis.get("operation"),
                        "confidence": array_analysis.get("confidence"),
                    })

        # Track A 3x: Minimal guidance only when explicitly lite (tests without profile keep full for backward compat)
        is_lite = (self.context.config.get("profile") == "lite") if self.context and "profile" in self.context.config else False
        compact_mode = (
            self._schema_router is not None
            and getattr(self._schema_router, "enabled", False)
            and getattr(self._schema_router, "compact_mode", False)
        )
        if is_lite:
            if compact_mode:
                tool_guidance = json.dumps({
                    "role": "system",
                        "content": (
                            "33k Tool: call_tool(tool,args). Tools: read(path) write(path,content) list(path) delete(path) done.\n"
                            "JSON: {\"tool_calls\":[{\"function\":{\"name\":\"call_tool\",\"arguments\":{\"tool\":\"write\",\"args\":{\"path\":\"a.txt\",\"content\":\"hi\"}}}}]}\n"
                            "ONE per turn. Check exists first. /math for math. TEMPLATE:todo:index.html expands to full file (use for todo app)."
                        )
                    }, ensure_ascii=False)
            else:
                tool_guidance = json.dumps({
                    "role": "system",
                    "content": (
                        "33k Tools: write_file(path,content) read_file(path) list_directory\n"
                        "JSON: {\"tool_calls\":[{\"function\":{\"name\":\"write_file\",\"arguments\":{\"path\":\"a.txt\",\"content\":\"hi\"}}}]} "
                        "ONE per turn. Check exists first. /math for math. TEMPLATE:todo:index.html expands to full file (use for todo app)."
                    )
                }, ensure_ascii=False)
        else:
            tool_guidance = json.dumps({
                 "role": "system",
                 "content": "You are a tool-using agent. You have access to the following tools:\n" +
                 "\n".join([f"- {s['function']['name']}: {s['function'].get('description', '')}" for s in self._tool_schemas]) +
                 "\n\nTool schemas:\n" +
                 "\n".join([json.dumps(s['function'], ensure_ascii=False) for s in self._tool_schemas]) +
                 "\n\nRULES:\n" +
                 "1. When you need to use a tool, respond with ONLY a JSON object in this exact format:\n" +
                 '{"tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "tool_name", "arguments": {"param1": "value1", "param2": "value2"}}}]}\n' +
                 "2. Call EXACTLY ONE tool per response. Do not call multiple tools in a single response.\n" +
                 "3. Do not include any other text, markdown, or code blocks in your response when calling a tool.\n" +
                 "4. For search/find tasks: use list_directory or read_file to check existence. If a file does not exist, report that it does not exist. Do NOT create the file.\n" +
                 "5. After tool results are provided, you may continue the conversation or call another tool if needed.\n" +
                 "6. For mathematical problems (algebra, calculus, limits, matrices), the /math command can be used for exact symbolic computation. Use tools for non-math tasks only.\n" +
                 "7. For simple chat/greetings (e.g., 'Say hello'), respond directly with text and do NOT use tools."
             }, ensure_ascii=False)
        self._system_prompt = tool_guidance
        self.context.append_message("system", tool_guidance)
        if not is_lite and self.context is not None:
            self.context.events.emit("system.message", {"content": tool_guidance})

        session_id = self._get_session_id()
        for round_idx in range(self.max_rounds):
            self._round = round_idx + 1
            self.context.check_cancelled()
            # P1 FIX: Unified single pruner (was dual: Summarizer.fold + prune). Now single ContextPruner handles both
            # token budget and message count (per-model via core.context calibration), preserves assistant tool_calls
            needs_prune = False
            est_tokens = Summarizer.estimate_tokens(str(self.context.messages))
            if est_tokens > self._token_budget or len(self.context.messages) > self._context_pruner.max_messages:
                needs_prune = True
            if needs_prune:
                pruned = self._context_pruner.prune(self.context.messages, task_state)
                self.context.messages = pruned.messages
                if self.context is not None:
                    self.context.events.emit("context.pruned", {
                        "removed_count": pruned.removed_count,
                        "estimated_tokens_before": pruned.estimated_tokens_before,
                        "estimated_tokens_after": pruned.estimated_tokens_after,
                        "strategy": pruned.strategy,
                        "estimated_tokens": est_tokens,
                    })

            if self._context_builder is not None:
                session_id = self._get_session_id()
                built = self._context_builder.build(session_id, user_text)
                memory_context = built.get("memory", "")
                if memory_context:
                    memory_msg = json.dumps({"role": "system", "content": memory_context}, ensure_ascii=False)
                    self.context.append_message("system", memory_msg)
                    if self.context is not None:
                        self.context.events.emit("memory.augmented", {"session_id": session_id, "context_length": len(memory_context)})
                    self.context.events.emit(SYSTEM_MESSAGE, {"content": memory_context, "provenance": "memory"})

            # P0 FIX: Removed duplicate turn.start emit (was firing per-round + outer). Use turn.round for per-round.
            active_schemas = self._get_active_tool_schemas()
            self.context.events.emit("turn.round", {
                "user_text": user_text,
                "session_id": session_id,
                "round": round_idx,
                "tools_available": [s["function"]["name"] for s in active_schemas],
            })

            # P1 FIX: Harden prompt injections — inject as user with prefix, not system (prevents privilege escalation)
            for injection in list(self.context.prompt_injections):
                safe_content = f"[injected context] {injection.content}"
                self.context.append_message("user", safe_content)
                if self.context is not None:
                    self.context.events.emit(USER_MESSAGE, {"content": safe_content, "provenance": "injection"})
            self.context.prompt_injections.clear()

            # Inject array guidance if facts changed (bounded, controlled API)
            if self._array_helper is not None and array_facts_changed and array_facts:
                guidance = self._array_helper.build_guidance(array_facts)
                if guidance:
                    self.context.prompt_injections.append(
                        Message("user", f"[array context] {guidance}")
                    )

            if is_lite:
                messages = self._compile_request_envelope(session_id, tool_guidance).messages
            else:
                messages = self.context.messages
            if self.stream:
                chunks = model.stream_chat(messages, active_schemas)
                response = self._consume_stream(chunks, on_stream=on_stream)
            else:
                response = model.chat(messages, active_schemas)

            tool_calls = response.tool_calls or []
            # main.py registers OllamaToolCallParser under its own name
            # ("ollama_tool_call_parser"); accept the generic role name too.
            parser = self.context.plugins.get("ollama_tool_call_parser") or self.context.plugins.get("tool_call_parser")
            if not tool_calls and parser is not None and response.content:
                tool_calls = parser.parse(response, active_schemas) or []

            # Expand compact-schema logical calls (call_tool) into real tool calls
            if tool_calls:
                tool_calls = self._expand_compact_calls(tool_calls)

            # P0 FIX: Filter out duplicate-successful tool calls BEFORE appending assistant message.
            # This prevents small models from copying their own prior tool calls in the context.
            duplicate_filtered = False
            if tool_calls:
                filtered_calls = []
                for call in tool_calls:
                    sig = self._sig(call)
                    if sig not in self._successful_calls:
                        filtered_calls.append(call)

                if not filtered_calls:
                    # All tool calls are duplicates of previously successful ones.
                    # Force a text-only assistant message and strong guidance, then continue
                    # to the next round so the model can reply with text instead of tools.
                    duplicate_filtered = True
                    assistant_content = response.content or "Task already completed."

                    guidance = json.dumps(
                        {
                            "role": "system",
                            "content": (
                                "You already completed the requested task successfully. "
                                "Do not call any tools again. Respond with a short text message only."
                            ),
                        },
                        ensure_ascii=False,
                    )
                    self.context.append_message("system", guidance)
                    if self.context is not None:
                        self.context.events.emit("system.message", {"content": guidance})

                    # Skip tool execution for this round; model should respond with text on next round.
                    tool_calls = []
                else:
                    tool_calls = filtered_calls
                    assistant_content = response.content
            else:
                assistant_content = response.content

            self.context.append_message("assistant", assistant_content, tool_calls=tool_calls)
            if self.context is not None:
                self.context.events.emit("assistant.message", {"content": assistant_content, "tool_calls": tool_calls})

            if duplicate_filtered:
                # Skip tool execution for this round; model should respond with text on next round.
                continue

            if not tool_calls and self._parse_retry_count < 2 and response.content:
                looks_like_tool_use = any(name.lower() in response.content.lower() for name in [s["function"]["name"] for s in self._tool_schemas])
                if looks_like_tool_use:
                    self._parse_retry_count += 1
                    guidance = json.dumps({
                        "role": "system",
                        "content": "You did not emit a tool call. You MUST respond with a JSON object containing tool_calls. Available tools: " +
                        ", ".join([s["function"]["name"] for s in self._tool_schemas]) +
                        ". Example: {\"tool_calls\": [{\"id\": \"call_1\", \"type\": \"function\", \"function\": {\"name\": \"write_file\", \"arguments\": {\"path\": \"a.txt\", \"content\": \"hello\"}}}]}"
                    }, ensure_ascii=False)
                    self.context.append_message("system", guidance)
                    if self.context is not None:
                        self.context.events.emit("system.message", {"content": guidance})
                    continue

            if not tool_calls:
                # App Completion Verifier gate: when the model declares completion
                # (text response, no tool calls), run deterministic verification before
                # allowing the agent to return. If verification fails, inject feedback
                # and continue to the next round.
                if self._app_verifier is not None:
                    # Define criteria on first completion attempt
                    if not task_state.get("verifier_criteria_defined"):
                        self._app_verifier.define_criteria(user_text, task_state)
                        task_state["verifier_criteria_defined"] = True

                    workspace = self.context.config.get("workspace", ".") if self.context else "."
                    all_passed = self._app_verifier.verify_completion(workspace, task_state)

                    if not all_passed:
                        feedback = self._app_verifier.get_feedback()
                        # Inject as user message per injection-hardening invariant
                        self.context.append_message("user", f"[verification feedback] {feedback}")
                        if self.context is not None:
                            self.context.events.emit("verification.failed", {
                                "session_id": session_id,
                                "failed_count": len([r for r in self._app_verifier._results if not r.passed]),
                            })
                        continue  # Keep working — don't return yet

                    # All criteria passed
                    if self.context is not None:
                        self.context.events.emit("verification.passed", {
                            "session_id": session_id,
                            "total_criteria": len(self._app_verifier._criteria),
                        })

                # Mark session outcome for training data collection
                self._mark_session_outcome(user_text, session_id, task_state)

                self.context.events.emit("turn.end", {"final_result": response.content, "session_id": session_id})
                return response.content

            failed_this_round: list[str] = []
            successful_results: list[str] = []
            for call in tool_calls:
                self.context.check_cancelled()
                fn = call.get("function", {})
                if not isinstance(fn, dict):
                    continue
                name = str(fn.get("name", ""))
                args = fn.get("arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                call_sig = self._sig(call)

                if self._blocked(call_sig):
                    result = json.dumps({
                        "error": f"Duplicate failed call detected: {name} with same arguments. Aborting to prevent infinite loop.",
                        "tool": name,
                        "arguments": args,
                    }, ensure_ascii=False)
                    self.context.append_message("tool", result, tool_name=name)
                    self.context.events.emit("tool.result", {
                        "tool": name,
                        "arguments": args,
                        "result": result,
                        "success": False,
                    })
                    failed_this_round.append(name)
                    continue

                # 3x 100% success: block duplicate successful calls (prevents 1.5B loop on 'Say hello' -> write_file hello.txt repeatedly)
                if call_sig in self._successful_calls:
                    result = json.dumps({
                        "error": f"Tool '{name}' already succeeded with same arguments. Do not repeat.",
                        "tool": name,
                        "arguments": args,
                        "duplicate": True,
                    }, ensure_ascii=False)
                    self.context.append_message("tool", result, tool_name=name)
                    self.context.events.emit("tool.result", {
                        "tool": name,
                        "arguments": args,
                        "result": result,
                        "success": False,
                    })
                    # Guide model to chat instead of re-hallucinating tool
                    self.context.append_message("system", json.dumps({"role":"system","content": f"Tool '{name}' already succeeded. Do not call it again with same arguments. For simple chat like 'Say hello', respond directly without tools."}, ensure_ascii=False))
                    failed_this_round.append(name)
                    continue

                try:
                    start_time = time.perf_counter()
                    # Optional ArrayHelper action review for write-related tools
                    if self._array_helper is not None and name == "write_file" and array_facts:
                        review = self._array_helper.review_action(name, args, array_facts)
                        if review["status"] == "warn" and self.context is not None:
                            self.context.events.emit("array.action.reviewed", {
                                "session_id": session_id,
                                "status": "warn",
                                "reason": review["reason"],
                            })
                            # Inject bounded warning guidance
                            warning = f"Array helper: {review['reason']}"
                            self.context.prompt_injections.append(
                                Message("user", f"[array context] {warning}")
                            )

                    result = self._execute_tool_call(call)
                    duration_ms = (time.perf_counter() - start_time) * 1000.0
                    self._log_tool_decision(name, args, result, success=True, duration_ms=duration_ms)
                except Exception as exc:
                    result = json.dumps({"error": str(exc), "tool": name, "arguments": args}, ensure_ascii=False)
                    failed_this_round.append(name)
                    self._log_tool_decision(name, args, result, success=False, error=str(exc))
                    # Guide model away from non-existent tools after first failure
                    if "Unknown tool" in str(exc):
                        guidance = json.dumps({
                            "role": "system",
                            "content": f"Tool '{name}' does not exist. Respond with text instead of tools.",
                        }, ensure_ascii=False)
                        self.context.append_message("system", guidance)
                        if self.context is not None:
                            self.context.events.emit("system.message", {"content": guidance})
                    recovery = self.context.plugins.get("error_recovery")
                    if recovery is not None:
                        failure_type = self._classify_failure(exc, {"tool_name": name, "arguments": args}).value
                        action = recovery.handle_failure(failure_type, {
                            "tool_name": name,
                            "arguments": args,
                            "result": result,
                        })
                        if action.action == "retry":
                            try:
                                result = self._execute_tool_call(call)
                                failed_this_round.remove(name)
                            except Exception:
                                pass
                        elif action.action == "fallback":
                            result = json.dumps({
                                "error": f"{name} failed after retries. Fallback not yet implemented.",
                                "tool": name,
                                "arguments": args,
                             }, ensure_ascii=False)
                    self.context.append_message("tool", result, tool_name=name)
                    self.context.events.emit("tool.result", {
                        "tool": name,
                        "arguments": args,
                        "result": result,
                        "success": False,
                    })
                else:
                    self._successful_calls.add(call_sig)
                    # Compress tool results for compact-schema mode to save context tokens
                    if self._schema_router is not None and getattr(self._schema_router, "compact_mode", False):
                        result = self._schema_router.compress_result(result)
                    self.context.append_message("tool", result, tool_name=name)
                    if name not in task_state["tools_used"]:
                        task_state["tools_used"].append(name)
                    if name == "write_file":
                        path = args.get("path", "")
                        if path and path not in task_state["files_touched"]:
                            task_state["files_touched"].append(path)

                    self.context.events.emit("tool.result", {
                        "tool": name,
                        "arguments": args,
                        "result": result,
                        "success": True,
                    })
                    successful_results.append(result)

                    # Optional ArrayHelper context analysis after read_file
                    if self._array_helper is not None and name == "read_file":
                        context_analysis = self._array_helper.analyze_context(
                            result[:self._array_helper._MAX_EXCERPT_LENGTH]
                        )
                        if context_analysis.get("confidence") in ("medium", "high"):
                            merged_facts = {**array_facts, **context_analysis}
                            if self._array_helper.update_facts(merged_facts):
                                array_facts = merged_facts
                                array_facts_changed = True
                                if self.context is not None:
                                    self.context.events.emit("array.context.updated", {
                                        "session_id": session_id,
                                        "representation": context_analysis.get("representation"),
                                        "shape": context_analysis.get("shape"),
                                        "confidence": context_analysis.get("confidence"),
                                    })

            # Pre-existing: continue until model emits no tool_calls (not auto-done after first write) — enables multi-file app builds within 33k
            if len(failed_this_round) > 1 and self._replan_count < 2:
                replan_msg = RepairMessageBuilder.global_replan(failed_this_round)
                self.context.append_message("system", replan_msg)
                if self.context is not None:
                    self.context.events.emit("replan", {"failed_tools": failed_this_round, "replan_count": self._replan_count})

        self.context.events.emit("turn.end", {"final_result": "", "error": "max_rounds_exceeded", "session_id": session_id})

        # Mark session failure for training data collection
        if self.context is not None:
            self._mark_session_outcome_failure(user_text, session_id, task_state)

        raise ToolError(f"Agent exceeded maximum tool-call rounds ({self.max_rounds}).")

    def _mark_session_outcome(self, user_text: str, session_id: str, task_state: dict[str, Any]) -> None:
        """Mark the session outcome for training data collection.

        Called when the agent successfully completes a task. If AppVerifier
        is present and passed, outcome is "success". Otherwise "success"
        (no verifier = trust model completion).
        """
        if self.context is None:
            return
        event_logger = self.context.plugins.get("event_logger")
        if event_logger is None or not hasattr(event_logger, "mark_session_outcome"):
            return

        # Determine app type from user request
        request_lower = user_text.lower()
        if any(kw in request_lower for kw in {"todo", "task list"}):
            app_type = "todo"
        elif any(kw in request_lower for kw in {"auth", "login", "signup"}):
            app_type = "auth"
        elif any(kw in request_lower for kw in {"crud", "api", "endpoint"}):
            app_type = "crud"
        elif any(kw in request_lower for kw in {"calculat", "math", "sum", "total"}):
            app_type = "calculator"
        elif any(kw in request_lower for kw in {"dashboard", "chart"}):
            app_type = "dashboard"
        else:
            app_type = "generic"

        metadata: dict[str, Any] = {
            "files_created": task_state.get("files_touched", []),
            "tools_used": task_state.get("tools_used", []),
            "app_type": app_type,
            "model_turns": self._round,
        }
        event_logger.mark_session_outcome("success", metadata)

    def _mark_session_outcome_failure(self, user_text: str, session_id: str, task_state: dict[str, Any]) -> None:
        """Mark the session as failed for training data collection.

        Called when the agent fails to complete a task (e.g., max rounds exceeded).
        """
        if self.context is None:
            return
        event_logger = self.context.plugins.get("event_logger")
        if event_logger is None or not hasattr(event_logger, "mark_session_outcome"):
            return

        request_lower = user_text.lower()
        if any(kw in request_lower for kw in {"todo", "task list"}):
            app_type = "todo"
        elif any(kw in request_lower for kw in {"auth", "login", "signup"}):
            app_type = "auth"
        elif any(kw in request_lower for kw in {"crud", "api", "endpoint"}):
            app_type = "crud"
        elif any(kw in request_lower for kw in {"calculat", "math", "sum", "total"}):
            app_type = "calculator"
        elif any(kw in request_lower for kw in {"dashboard", "chart"}):
            app_type = "dashboard"
        else:
            app_type = "generic"

        metadata: dict[str, Any] = {
            "files_created": task_state.get("files_touched", []),
            "tools_used": task_state.get("tools_used", []),
            "app_type": app_type,
            "model_turns": self._round,
            "failure_reason": "max_rounds_exceeded",
        }
        event_logger.mark_session_outcome("failure", metadata)
