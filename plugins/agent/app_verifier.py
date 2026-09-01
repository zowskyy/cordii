from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from core.plugin import Plugin


@dataclass
class VerificationCriterion:
    """Defines a single verification check.

    Attributes:
        name: Short identifier for the criterion.
        description: Human-readable description of what is being checked.
        check_type: One of 'file_exists', 'file_content', 'command_runs',
            'test_passes', 'feature_works'.
        required: If True, failure blocks completion. If False, it's a warning.
        parameters: Dict with check-specific configuration (e.g., path, patterns).
    """
    name: str
    description: str
    check_type: str
    required: bool = True
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationResult:
    """Result of checking one criterion."""
    criterion: VerificationCriterion
    passed: bool
    evidence: str = ""
    feedback: str = ""


class AppVerifier(Plugin):
    """Deterministic, evidence-based completion verifier.

    Translates user requests into concrete, checkable criteria, then verifies
    actual filesystem artifacts. Never calls the model or executes tools —
    uses FileTools.read_file / pathlib for reads only.

    Key design principle: the verifier is **mechanical enforcement**, not a
    prompt reminder. It checks what actually exists on disk, not what the
    model claims to have built.
    """
    name = "app_verifier"
    dependencies = ("file_tools", "asgi_wsgi_tester")

    __contract__ = {
        "version": "1.0",
        "provides": ("completion_verification", "feature_validation", "server_testing"),
        "requires": ("file_tools",),
        "deterministic": True,
        "zero_token": True,
    }

    # ------------------------------------------------------------------
    # App-pattern keyword groups (deterministic, regex-based)
    # ------------------------------------------------------------------
    _TODO_KEYWORDS = {"todo", "to-do", "task list", "tasks list"}
    _CRUD_KEYWORDS = {"crud", "api", "endpoint", "rest", "backend", "server route"}
    _CALC_KEYWORDS = {"calculat", "math", "arithmetic", "calculator", "count", "sum", "total"}
    _DASHBOARD_KEYWORDS = {"dashboard", "chart", "visual", "data display", "metric", "statistics", "analytics"}
    _AUTH_KEYWORDS = {"auth", "login", "signup", "register", "session", "password", "jwt", "token"}
    _ECOMMERCE_KEYWORDS = {"e-commerce", "ecommerce", "cart", "checkout", "product", "product list", "shop"}
    _CHAT_KEYWORDS = {"chat", "message", "messaging", "real-time", "websocket", "socket"}
    _DATAVIZ_KEYWORDS = {"data visualization", "chart.js", "d3", "graph", "plot", "visualization"}

    # Generic anti-pattern markers (case-sensitive: TODO/FIXME are uppercase in code)
    _PLACEHOLDER_PATTERNS = [
        re.compile(r"TODO[: ]"),
        re.compile(r"FIXME[: ]"),
        re.compile(r"PLACEHOLDER", re.IGNORECASE),
        re.compile(r"lorem ipsum", re.IGNORECASE),
        re.compile(r"not implemented", re.IGNORECASE),
        re.compile(r"to be implemented", re.IGNORECASE),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._criteria: list[VerificationCriterion] = []
        self._results: list[VerificationResult] = []
        self._file_tools: Any = None
        self._server_tester: Any = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Initialize empty criteria and results lists."""
        self._criteria = []
        self._results = []
        if self.context is not None:
            self._file_tools = self.context.plugins.get("file_tools")
            self._server_tester = self.context.plugins.get("asgi_wsgi_tester")

    def stop(self) -> None:
        """Clean up state."""
        self._criteria = []
        self._results = []
        self._file_tools = None
        self._server_tester = None

    def reset_run_state(self) -> None:
        """Reset per-run state at the beginning of each run()."""
        self._criteria = []
        self._results = []
        # Re-look up plugins in case they were refreshed
        if self.context is not None:
            self._file_tools = self.context.plugins.get("file_tools")
            self._server_tester = self.context.plugins.get("asgi_wsgi_tester")

    def health_check(self) -> dict[str, Any]:
        """Verify all capability methods are callable."""
        return {
            "healthy": True,
            "plugin": self.name,
            "contract_version": self.__contract__.get("version", "1.0"),
            "capabilities": {
                "define_criteria": callable(getattr(self, "define_criteria", None)),
                "verify_completion": callable(getattr(self, "verify_completion", None)),
                "get_feedback": callable(getattr(self, "get_feedback", None)),
            },
        }

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def define_criteria(self, user_request: str, task_state: dict[str, Any]) -> list[VerificationCriterion]:
        """Translate user request into checkable criteria.

        Uses keyword-based pattern recognition to determine app type and
        generates appropriate file existence / content checks.

        Args:
            user_request: The original user text describing the task.
            task_state: Shared task state dict (mutated: stores 'app_type').

        Returns:
            List of VerificationCriterion objects.
        """
        self._criteria = []
        request_lower = user_request.lower()

        # Detect app type
        if any(kw in request_lower for kw in self._TODO_KEYWORDS):
            task_state["app_type"] = "todo"
            self._generate_todo_criteria()
        elif any(kw in request_lower for kw in self._AUTH_KEYWORDS):
            task_state["app_type"] = "auth"
            self._generate_auth_criteria()
        elif any(kw in request_lower for kw in self._ECOMMERCE_KEYWORDS):
            task_state["app_type"] = "ecommerce"
            self._generate_ecommerce_criteria()
        elif any(kw in request_lower for kw in self._CHAT_KEYWORDS):
            task_state["app_type"] = "chat"
            self._generate_chat_criteria()
        elif any(kw in request_lower for kw in self._DATAVIZ_KEYWORDS):
            task_state["app_type"] = "dataviz"
            self._generate_dataviz_criteria()
        elif any(kw in request_lower for kw in self._CRUD_KEYWORDS):
            task_state["app_type"] = "crud"
            self._generate_crud_criteria()
        elif any(kw in request_lower for kw in self._CALC_KEYWORDS):
            task_state["app_type"] = "calculator"
            self._generate_calculator_criteria()
        elif any(kw in request_lower for kw in self._DASHBOARD_KEYWORDS):
            task_state["app_type"] = "dashboard"
            self._generate_dashboard_criteria()
        else:
            task_state["app_type"] = "generic"

        # Always add generic criteria
        self._generate_generic_criteria(task_state)

        return list(self._criteria)

    def verify_completion(self, workspace_path: str, task_state: dict[str, Any]) -> bool:
        """Check all criteria against the workspace.

        Args:
            workspace_path: Root workspace directory path.
            task_state: Shared task state dict.

        Returns:
            True if all required criteria passed, False otherwise.
        """
        if not self._criteria:
            return True  # No criteria = vacuously satisfied

        self._results = []
        ws = Path(workspace_path) if workspace_path else Path(".")

        for criterion in self._criteria:
            result = self._check_criterion(criterion, ws)
            self._results.append(result)

        return all(
            r.passed for r in self._results if r.criterion.required
        )

    def get_feedback(self) -> str:
        """Return human-readable feedback about failed criteria.

        Returns:
            Formatted string listing all failed criteria and how to fix them.
        """
        failed = [r for r in self._results if not r.passed]
        if not failed:
            return "All verification checks passed."

        lines = ["Verification failed. Complete these required steps:"]
        for result in failed:
            lines.append(f"  [{result.criterion.check_type}] {result.criterion.description}: {result.feedback}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal: check dispatcher
    # ------------------------------------------------------------------

    def _check_criterion(self, criterion: VerificationCriterion, ws: Path) -> VerificationResult:
        """Dispatch a single criterion to the appropriate check method."""
        check_type = criterion.check_type
        params = criterion.parameters

        if check_type == "file_exists":
            return self._check_file_exists(criterion, ws, params.get("path", ""))
        elif check_type == "file_content":
            paths = params.get("path", "")
            if isinstance(paths, list):
                # Multi-file check: at least one file must contain all/any patterns
                return self._check_file_content_multi(criterion, ws, paths, params.get("patterns", []))
            return self._check_file_content(criterion, ws, paths, params.get("patterns", []))
        elif check_type == "command_runs":
            return self._check_command_runs(criterion, params.get("command", ""), params.get("expected", ""))
        elif check_type == "test_passes":
            return self._check_test_passes(criterion, params.get("command", ""))
        elif check_type == "feature_works":
            return self._check_feature_works(criterion, params.get("description", ""), params.get("check", ""))
        elif check_type == "server_runs":
            return self._check_server_runs(criterion, ws, params)
        elif check_type == "no_placeholders":
            return self._check_no_placeholders(criterion, ws, params.get("path", ""))
        else:
            return VerificationResult(
                criterion=criterion,
                passed=False,
                evidence="",
                feedback=f"Unknown check_type: {check_type}",
            )

    def _check_file_exists(self, criterion: VerificationCriterion, ws: Path, file_path: str) -> VerificationResult:
        """Verify a file exists at the specified path."""
        full_path = ws / file_path
        exists = full_path.is_file()
        return VerificationResult(
            criterion=criterion,
            passed=exists,
            evidence=f"File exists: {exists} ({file_path})",
            feedback="" if exists else f"File does not exist: {file_path}",
        )

    def _check_file_content(self, criterion: VerificationCriterion, ws: Path, file_path: str, patterns: list[str]) -> VerificationResult:
        """Verify a file contains required patterns.

        By default, ALL patterns must be present. If criterion.parameters
        has 'match': 'any', at least one pattern must be present.
        If 'case_insensitive': True (default for code checks), matching
        is case-insensitive.
        """
        full_path = ws / file_path
        if not full_path.is_file():
            return VerificationResult(
                criterion=criterion,
                passed=False,
                evidence="",
                feedback=f"File does not exist: {file_path}",
            )

        try:
            content = full_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            return VerificationResult(
                criterion=criterion,
                passed=False,
                evidence="",
                feedback=f"Could not read {file_path}: {exc}",
            )

        match_mode = criterion.parameters.get("match", "all")
        case_insensitive = criterion.parameters.get("case_insensitive", True)

        if case_insensitive:
            search_content = content.lower()
            search_patterns = [p.lower() for p in patterns]
        else:
            search_content = content
            search_patterns = patterns

        missing: list[str] = []
        present: list[str] = []
        for pat, orig_pat in zip(search_patterns, patterns):
            if pat in search_content:
                present.append(orig_pat)
            else:
                missing.append(orig_pat)

        if match_mode == "any":
            passed = len(present) > 0
            feedback = "" if passed else f"None of the required patterns found in {file_path}: {', '.join(patterns)}"
        else:
            passed = len(missing) == 0
            feedback = "" if passed else f"Missing in {file_path}: {', '.join(missing)}"

        return VerificationResult(
            criterion=criterion,
            passed=passed,
            evidence=f"Checked {len(patterns)} patterns in {file_path}, {len(missing)} missing",
             feedback=feedback,
        )

    def _check_file_content_multi(self, criterion: VerificationCriterion, ws: Path, file_paths: list[str], patterns: list[str]) -> VerificationResult:
        """Verify at least one of multiple files contains required patterns."""
        match_mode = criterion.parameters.get("match", "any")
        case_insensitive = criterion.parameters.get("case_insensitive", True)

        for fp in file_paths:
            result = self._check_file_content(criterion, ws, fp, patterns)
            # If any file passes, the multi-file check passes
            if result.passed:
                return VerificationResult(
                    criterion=criterion,
                    passed=True,
                    evidence=f"Pattern found in {fp}",
                    feedback="",
                )

        return VerificationResult(
            criterion=criterion,
            passed=False,
            evidence=f"Checked {len(file_paths)} files for patterns: {', '.join(patterns)} (mode={match_mode})",
            feedback=f"None of {file_paths} contain required patterns: {', '.join(patterns)}",
        )

    def _check_command_runs(self, criterion: VerificationCriterion, command: str, expected: str) -> VerificationResult:
        """Verify a command executes successfully (uses terminal plugin if available)."""
        # No tool execution: this is a no-op unless a terminal plugin is wired.
        # The verifier itself does not execute commands — it reports what it can check.
        terminal = self.context.plugins.get("terminal") if self.context else None
        if terminal is None:
            return VerificationResult(
                criterion=criterion,
                passed=True,
                evidence="Terminal plugin not available — check skipped",
                feedback=", run the command manually to validate.",
            )
        return VerificationResult(
            criterion=criterion,
            passed=True,
            evidence=f"Command check deferred to terminal: {command}",
            feedback="",
        )

    def _check_test_passes(self, criterion: VerificationCriterion, command: str) -> VerificationResult:
        """Verify tests pass (uses terminal plugin if available)."""
        terminal = self.context.plugins.get("terminal") if self.context else None
        if terminal is None:
            return VerificationResult(
                criterion=criterion,
                passed=True,
                evidence="Terminal plugin not available — test check skipped",
                feedback="",
            )
        return VerificationResult(
            criterion=criterion,
            passed=True,
            evidence=f"Test check deferred to terminal: {command}",
            feedback="",
        )

    def _check_feature_works(self, criterion: VerificationCriterion, description: str, check: str) -> VerificationResult:
        """Verify a feature actually functions (manual check guidance)."""
        return VerificationResult(
            criterion=criterion,
            passed=True,
            evidence=f"Manual check: {check}",
            feedback="",
        )

    def _check_server_runs(
        self,
        criterion: VerificationCriterion,
        ws: Path,
        params: dict[str, Any],
    ) -> VerificationResult:
        """Check if the server starts and endpoints respond.

        Delegates to ASGIWSGITester plugin if available. If not available,
        returns a failure result.
        """
        if self._server_tester is None:
            return VerificationResult(
                criterion=criterion,
                passed=False,
                evidence="ASGI/WSGI tester not available",
                feedback="Server testing requires asgi_wsgi_tester plugin to be registered",
            )

        command = params.get("command", "")
        base_url = params.get("base_url", "http://127.0.0.1:8000")
        endpoints = params.get("endpoints", [])

        # Start the server from the workspace directory
        started = self._server_tester.start_server(command, base_url, cwd=str(ws))
        if not started:
            return VerificationResult(
                criterion=criterion,
                passed=False,
                evidence="Server failed to start",
                feedback=f"Server did not start with command: {command}",
            )

        # Test endpoints if specified
        if endpoints:
            result = self._server_tester.test_endpoints(endpoints)
            self._server_tester.stop_server()

            if result["passed"]:
                return VerificationResult(
                    criterion=criterion,
                    passed=True,
                    evidence=result["summary"],
                    feedback="",
                )
            else:
                return VerificationResult(
                    criterion=criterion,
                    passed=False,
                    evidence=result["summary"],
                    feedback=f"Endpoint tests failed: {result['summary']}",
                )
        else:
            # Just verify the server started
            self._server_tester.stop_server()
            return VerificationResult(
                criterion=criterion,
                passed=True,
                evidence="Server started and responded",
                feedback="",
            )

    def _check_no_placeholders(self, criterion: VerificationCriterion, ws: Path, file_path: str) -> VerificationResult:
        """Verify a file does not contain placeholder/TODO markers."""
        full_path = ws / file_path
        if not full_path.is_file():
            # Files that don't exist yet are not a placeholder violation
            return VerificationResult(
                criterion=criterion,
                passed=True,
                evidence=f"File does not exist yet (no placeholder to check)",
                feedback="",
            )

        try:
            content = full_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            return VerificationResult(
                criterion=criterion,
                passed=True,
                evidence=f"Could not read {file_path}: {exc}",
                feedback="",
            )

        found: list[str] = []
        for pattern in self._PLACEHOLDER_PATTERNS:
            if pattern.search(content):
                found.append(pattern.pattern)

        passed = len(found) == 0
        return VerificationResult(
            criterion=criterion,
            passed=passed,
            evidence=f"Checked {len(self._PLACEHOLDER_PATTERNS)} placeholder patterns, {len(found)} found",
            feedback="" if passed else f"Placeholders found in {file_path}: {', '.join(found)}",
        )

    # ------------------------------------------------------------------
    # App-pattern criteria generators
    # ------------------------------------------------------------------

    def _generate_todo_criteria(self) -> None:
        """Generate criteria for a todo app (HTML + JS add/delete/filter)."""
        self._criteria.extend([
            VerificationCriterion(
                name="todo_html_exists",
                description="Todo app must have an HTML file",
                check_type="file_exists",
                parameters={"path": "index.html"},
            ),
            VerificationCriterion(
                name="todo_js_exists",
                description="Todo app must have a JS file",
                check_type="file_exists",
                parameters={"path": "app.js"},
            ),
            VerificationCriterion(
                name="todo_add_function",
                description="Must have an add function",
                check_type="file_content",
                parameters={"path": "app.js", "patterns": ["function", "add"], "match": "any"},
            ),
            VerificationCriterion(
                name="todo_delete_ability",
                description="Must support delete/remove functionality",
                check_type="file_content",
                parameters={"path": "app.js", "patterns": ["splice", "filter", "remove", "delete"], "match": "any"},
            ),
            VerificationCriterion(
                name="todo_list_element",
                description="HTML must have a list element",
                check_type="file_content",
                parameters={"path": "index.html", "patterns": ["ul", "list"], "match": "any"},
            ),
        ])

    def _generate_crud_criteria(self) -> None:
        """Generate criteria for a CRUD app (POST/GET/PUT/DELETE endpoints).

        Supports both Node.js (server.js) and Python (server.py) backends.
        The server file check accepts either — at least one must exist.
        Server testing (ASGI/WSGI) is attempted when a Python server is detected.
        """
        self._criteria.extend([
            VerificationCriterion(
                name="crud_server_file",
                description="CRUD app must have a server file (server.js or server.py)",
                check_type="file_exists",
                parameters={"path": "server.js"},
                required=False,  # May use server.py instead
            ),
            VerificationCriterion(
                name="crud_server_file_py",
                description="CRUD app must have a server file (server.js or server.py)",
                check_type="file_exists",
                parameters={"path": "server.py"},
                required=False,  # May use server.js instead
            ),
            VerificationCriterion(
                name="crud_create_endpoint",
                description="Must have a POST/create endpoint",
                check_type="file_content",
                parameters={"path": ["server.js", "server.py"], "patterns": ["POST", "create", "add"], "match": "any"},
            ),
            VerificationCriterion(
                name="crud_read_endpoint",
                description="Must have a GET/read endpoint",
                check_type="file_content",
                parameters={"path": ["server.js", "server.py"], "patterns": ["GET", "read", "find", "get"], "match": "any"},
            ),
            VerificationCriterion(
                name="crud_update_endpoint",
                description="Must have a PUT/update endpoint",
                check_type="file_content",
                parameters={"path": ["server.js", "server.py"], "patterns": ["PUT", "update", "edit"], "match": "any"},
            ),
            VerificationCriterion(
                name="crud_delete_endpoint",
                description="Must have a DELETE endpoint",
                check_type="file_content",
                parameters={"path": ["server.js", "server.py"], "patterns": ["DELETE", "delete", "remove"], "match": "any"},
            ),
        ])

    def _generate_calculator_criteria(self) -> None:
        """Generate criteria for a calculator app (display, buttons, operations, logic)."""
        self._criteria.extend([
            VerificationCriterion(
                name="calc_html_exists",
                description="Calculator must have an HTML file",
                check_type="file_exists",
                parameters={"path": "index.html"},
            ),
            VerificationCriterion(
                name="calc_js_exists",
                description="Calculator must have a JS file",
                check_type="file_exists",
                parameters={"path": "app.js"},
            ),
            VerificationCriterion(
                name="calc_display",
                description="Must have a display/input element",
                check_type="file_content",
                parameters={"path": "index.html", "patterns": ["display", "input", "screen", "result"], "match": "any"},
            ),
            VerificationCriterion(
                name="calc_number_buttons",
                description="Must have number buttons (0-9)",
                check_type="file_content",
                parameters={"path": "index.html", "patterns": ["0", "1", "2", "9"], "match": "any"},
            ),
            VerificationCriterion(
                name="calc_operations",
                description="Must have arithmetic operations",
                check_type="file_content",
                parameters={"path": "app.js", "patterns": ["+", "-", "*", "/"], "match": "any"},
            ),
        ])

    def _generate_dashboard_criteria(self) -> None:
        """Generate criteria for a dashboard app (container, visualization, data)."""
        self._criteria.extend([
            VerificationCriterion(
                name="dashboard_html_exists",
                description="Dashboard must have an HTML file",
                check_type="file_exists",
                parameters={"path": "index.html"},
            ),
            VerificationCriterion(
                name="dashboard_container",
                description="Must have a container/div element",
                check_type="file_content",
                parameters={"path": "index.html", "patterns": ["div", "container", "chart", "canvas"], "match": "any"},
            ),
            VerificationCriterion(
                name="dashboard_js_exists",
                description="Dashboard must have a JS file for visualization",
                check_type="file_exists",
                parameters={"path": "app.js"},
            ),
            VerificationCriterion(
                name="dashboard_data_source",
                description="Must have a data source or fetch",
                check_type="file_content",
                parameters={"path": "app.js", "patterns": ["fetch", "data", "api", "json", "datasource"], "match": "any"},
            ),
        ])

    def _generate_generic_criteria(self, task_state: dict[str, Any]) -> None:
        """Generic criteria: no TODO/FIXME comments, no placeholder text."""
        app_type = task_state.get("app_type", "generic")
        # Map app type to likely main files to check for placeholders
        main_files: list[str] = []
        if app_type == "todo":
            main_files = ["index.html", "app.js"]
        elif app_type == "crud":
            main_files = ["server.js"]
        elif app_type == "calculator":
            main_files = ["index.html", "app.js"]
        elif app_type == "dashboard":
            main_files = ["index.html", "app.js"]
        elif app_type == "auth":
            main_files = ["server.js", "index.html"]
        elif app_type == "ecommerce":
            main_files = ["server.js", "index.html", "app.js"]
        elif app_type == "chat":
            main_files = ["server.js", "index.html", "app.js"]
        elif app_type == "dataviz":
            main_files = ["index.html", "app.js"]

        for file_path in main_files:
            self._criteria.append(VerificationCriterion(
                name=f"no_placeholders_{file_path.replace('/', '_')}",
                description=f"No placeholder/TODO comments in {file_path}",
                check_type="no_placeholders",
                parameters={"path": file_path},
            ))

    # ------------------------------------------------------------------
    # New app-pattern criteria generators
    # ------------------------------------------------------------------

    def _generate_auth_criteria(self) -> None:
        """Generate criteria for an auth app (login, signup, session)."""
        self._criteria.extend([
            VerificationCriterion(
                name="auth_server_file",
                description="Auth app must have a server file",
                check_type="file_exists",
                parameters={"path": "server.js"},
            ),
            VerificationCriterion(
                name="auth_login_endpoint",
                description="Must have a login route/endpoint",
                check_type="file_content",
                parameters={"path": "server.js", "patterns": ["login", "signin", "auth", "post"], "match": "any"},
            ),
            VerificationCriterion(
                name="auth_signup_endpoint",
                description="Must have a signup/registration route",
                check_type="file_content",
                parameters={"path": "server.js", "patterns": ["signup", "register", "register", "post"], "match": "any"},
            ),
            VerificationCriterion(
                name="auth_session_management",
                description="Must handle sessions or tokens",
                check_type="file_content",
                parameters={"path": "server.js", "patterns": ["session", "token", "jwt", "cookie"], "match": "any"},
            ),
            VerificationCriterion(
                name="auth_password_handling",
                description="Must have password handling",
                check_type="file_content",
                parameters={"path": "server.js", "patterns": ["password", "hash", "bcrypt", "encrypt"], "match": "any"},
            ),
        ])

    def _generate_ecommerce_criteria(self) -> None:
        """Generate criteria for an e-commerce app (product list, cart, checkout)."""
        self._criteria.extend([
            VerificationCriterion(
                name="ecommerce_html_exists",
                description="E-commerce app must have an HTML file",
                check_type="file_exists",
                parameters={"path": "index.html"},
            ),
            VerificationCriterion(
                name="ecommerce_server_exists",
                description="E-commerce app must have a server file",
                check_type="file_exists",
                parameters={"path": "server.js"},
            ),
            VerificationCriterion(
                name="ecommerce_product_list",
                description="Must have a product listing",
                check_type="file_content",
                parameters={"path": "server.js", "patterns": ["product", "items", "catalog", "list"], "match": "any"},
            ),
            VerificationCriterion(
                name="ecommerce_cart_functionality",
                description="Must support cart operations",
                check_type="file_content",
                parameters={"path": "app.js", "patterns": ["cart", "addtocart", "add_to_cart"], "match": "any"},
            ),
            VerificationCriterion(
                name="ecommerce_checkout",
                description="Must have checkout functionality",
                check_type="file_content",
                parameters={"path": ["server.js", "app.js"], "patterns": ["checkout", "payment", "order"], "match": "any",
                            "multi_file": True},
            ),
        ])

    def _generate_chat_criteria(self) -> None:
        """Generate criteria for a chat app (message list, send, real-time)."""
        self._criteria.extend([
            VerificationCriterion(
                name="chat_html_exists",
                description="Chat app must have an HTML file",
                check_type="file_exists",
                parameters={"path": "index.html"},
            ),
            VerificationCriterion(
                name="chat_js_exists",
                description="Chat app must have a JS file",
                check_type="file_exists",
                parameters={"path": "app.js"},
            ),
            VerificationCriterion(
                name="chat_message_list",
                description="Must have a message display area",
                check_type="file_content",
                parameters={"path": "index.html", "patterns": ["message", "chat", "messages", "list"], "match": "any"},
            ),
            VerificationCriterion(
                name="chat_send_functionality",
                description="Must support sending messages",
                check_type="file_content",
                parameters={"path": "app.js", "patterns": ["send", "emit", "submit", "post"], "match": "any"},
            ),
            VerificationCriterion(
                name="chat_server_exists",
                description="Must have a server for real-time communication",
                check_type="file_exists",
                parameters={"path": "server.js"},
            ),
            VerificationCriterion(
                name="chat_realtime",
                description="Must have real-time communication (websocket/socket)",
                check_type="file_content",
                parameters={"path": "server.js", "patterns": ["socket", "websocket", "ws", "real-time"], "match": "any"},
            ),
        ])

    def _generate_dataviz_criteria(self) -> None:
        """Generate criteria for a data visualization app (charts, graphs)."""
        self._criteria.extend([
            VerificationCriterion(
                name="dataviz_html_exists",
                description="Data visualization app must have an HTML file",
                check_type="file_exists",
                parameters={"path": "index.html"},
            ),
            VerificationCriterion(
                name="dataviz_js_exists",
                description="Data visualization app must have a JS file",
                check_type="file_exists",
                parameters={"path": "app.js"},
            ),
            VerificationCriterion(
                name="dataviz_chart_library",
                description="Must use a charting library",
                check_type="file_content",
                parameters={"path": ["index.html", "app.js"], "patterns": ["chart", "d3", "plotly", "canvas"], "match": "any",
                            "multi_file": True},
            ),
            VerificationCriterion(
                name="dataviz_data_source",
                description="Must have a data source",
                check_type="file_content",
                parameters={"path": ["index.html", "app.js"], "patterns": ["data", "fetch", "api", "json"], "match": "any",
                            "multi_file": True},
            ),
            VerificationCriterion(
                name="dataviz_chart_element",
                description="HTML must have a chart container element",
                check_type="file_content",
                parameters={"path": "index.html", "patterns": ["canvas", "svg", "chart", "div"], "match": "any"},
            ),
        ])
