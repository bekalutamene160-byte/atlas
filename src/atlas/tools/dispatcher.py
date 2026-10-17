"""
Tool dispatcher — the bridge between planner and tools.

When the planner emits a ToolCall, the dispatcher:
1. Looks up the tool by name (raises if unknown)
2. Validates arguments against the tool's args_model (rejects invalid)
3. Executes the tool (async)
4. Logs the call for tracing
5. Returns the ToolResult

Why this layer exists (why not call tools directly):
1. Validation happens BEFORE execution — invalid args never reach the tool
2. Unknown tools are caught (LLM might hallucinate a tool name)
3. Every call is logged — the trace shows what the agent did and when
4. The dispatcher is the single place to add cross-cutting concerns:
   - Rate limiting (don't call tools too fast)
   - Caching (don't re-execute identical calls)
   - Authorization (is this agent allowed to call this tool?)
   - Audit logging (for compliance)

Design decision: the dispatcher returns ToolResult, never raises.
If the tool name is unknown or args are invalid, it returns
ToolResult(success=False, error="..."). This lets the agent loop
handle failures gracefully instead of try/except around every dispatch.

Pro tip for advanced projects: at scale, add a "policy" layer between
the dispatcher and the tool. The policy checks:
- Is this agent role allowed to call this tool? (least privilege)
- Does this tool have side effects? (require approval)
- Has this exact call been made before? (return cached result)
ATLAS v1 has only read-only tools, so we skip the policy layer.
"""

from __future__ import annotations

from collections.abc import Sequence
import logging
from typing import Optional

from atlas.models import ToolCall, ToolResult
from atlas.tools.base import Tool

logger = logging.getLogger(__name__)


class ToolDispatcher:
    """Routes ToolCalls to the right Tool, with validation and logging."""

    def __init__(self, tools: Sequence[Tool]):
        """Initialize with a list of tools.

        Builds a registry: {tool.name: tool_instance}
        Duplicate names raise ValueError — each tool must be unique.
        """
        self.registry: dict[str, Tool] = {}
        for tool in tools:
            if tool.name in self.registry:
                raise ValueError(
                    f"Duplicate tool name: '{tool.name}'. "
                    f"Each tool must have a unique name."
                )
            self.registry[tool.name] = tool

        logger.info(f"ToolDispatcher initialized with {len(self.registry)} tools: "
                    f"{list(self.registry.keys())}")

    def get_tool_schemas(self) -> list[dict]:
        """Return schemas for all registered tools.

        Used by the planner to tell the LLM what tools are available.
        """
        return [tool.get_schema() for tool in self.registry.values()]

    async def dispatch(self, call: ToolCall) -> ToolResult:
        """Execute a tool call.

        Steps:
        1. Look up the tool (return error if unknown)
        2. Validate arguments (return error if invalid)
        3. Execute the tool (catch exceptions, return error)
        4. Log the result

        Never raises — always returns ToolResult.
        """
        # Step 1: Look up the tool
        tool: Optional[Tool] = self.registry.get(call.name)
        if tool is None:
            logger.warning(
                f"dispatch: unknown tool '{call.name}'. "
                f"Available: {list(self.registry.keys())}"
            )
            return ToolResult(
                success=False,
                data={},
                error=f"unknown_tool: '{call.name}' is not registered. "
                      f"Available tools: {list(self.registry.keys())}",
                latency_ms=0.0,
            )

        # Step 2: Validate arguments against the tool's schema
        # This catches malformed LLM output before it reaches the tool
        try:
            args = tool.args_model.model_validate(call.arguments)
        except Exception as e:
            logger.warning(
                f"dispatch: invalid args for '{call.name}': {e}. "
                f"Args: {call.arguments}"
            )
            return ToolResult(
                success=False,
                data={},
                error=f"invalid_arguments: {type(e).__name__}: {str(e)[:200]}",
                latency_ms=0.0,
            )

        # Step 3: Execute the tool
        # The tool's execute() catches its own exceptions, but we wrap
        # it as a safety net — the dispatcher must never crash the agent
        logger.info(f"dispatch: calling tool '{call.name}' with args: {call.arguments}")
        try:
            result: ToolResult = await tool.execute(args)
        except Exception as e:
            # This should never happen (tools catch their own exceptions)
            # but if it does, we don't crash the agent
            logger.error(
                f"dispatch: tool '{call.name}' raised unhandled exception: {e}",
                exc_info=True,
            )
            return ToolResult(
                success=False,
                data={},
                error=f"tool_exception: {type(e).__name__}: {str(e)[:200]}",
                latency_ms=0.0,
            )

        # Step 4: Log the result
        status = "ok" if result.success else "error"
        logger.info(
            f"dispatch: tool '{call.name}' returned {status} "
            f"(latency={result.latency_ms:.0f}ms)"
        )

        return result
