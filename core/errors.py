class HarnessError(Exception):
    """Base exception for the harness."""


class PluginError(HarnessError):
    """Plugin registration/lifecycle failure."""


class ToolError(HarnessError):
    """Tool execution failure."""


class ModelError(HarnessError):
    """Model/API failure."""


class WorkspaceError(ToolError):
    """Workspace path violation."""


class CancelledError(HarnessError):
    """Agent run was cancelled."""
