"""Tool registry: the single place where executable capabilities are declared.

Every tool registers with a name, a description and a safety level. The router
never calls a function directly — it looks the tool up here, and the safety
layer decides (based on the declared level) whether the call may proceed.
"""

from dataclasses import dataclass
from typing import Any, Callable

from app.schemas.commands import Intent, SafetyLevel

ToolHandler = Callable[..., dict[str, Any]]

# Which tool serves which router intent. Shared by the rule-based router and
# the LLM planner (which uses it to reject intent/tool mismatches).
INTENT_TOOL_MAP: dict[Intent, str] = {
    Intent.OPEN_FOLDER: "open_folder",
    Intent.OPEN_APP: "open_app",
    Intent.TYPE_TEXT: "type_text",
    Intent.SEARCH_WEB: "search_web",
    Intent.SPEAK: "speak",
}


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    safety_level: SafetyLevel
    handler: ToolHandler


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(
        self, name: str, description: str, safety_level: SafetyLevel
    ) -> Callable[[ToolHandler], ToolHandler]:
        """Decorator: register a function as a tool."""

        def decorator(handler: ToolHandler) -> ToolHandler:
            if name in self._tools:
                raise ValueError(f"tool already registered: {name}")
            self._tools[name] = Tool(name, description, safety_level, handler)
            return handler

        return decorator

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def all(self) -> list[Tool]:
        return sorted(self._tools.values(), key=lambda t: t.name)


registry = ToolRegistry()


def load_tools() -> None:
    """Import tool modules so their @registry.register decorators run."""
    from app.tools import (  # noqa: F401
        browser_tools,
        file_tools,
        text_tools,
        windows_tools,
    )
