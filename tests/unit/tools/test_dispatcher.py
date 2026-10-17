"""Unit tests for ToolDispatcher."""

from __future__ import annotations

import pytest

from atlas.models import ToolArgs, ToolCall, ToolResult
from atlas.tools.base import Tool
from atlas.tools.dispatcher import ToolDispatcher


# =============================================================================
# Test Fixtures: Mock Tools
# =============================================================================

class EchoArgs(ToolArgs):
    """Args for the echo tool."""
    message: str


class EchoTool(Tool):
    """A test tool that echoes the message back."""
    name = "echo"
    description = "Echo the message back."
    args_model = EchoArgs

    async def execute(self, args: EchoArgs) -> ToolResult:
        return ToolResult(
            success=True,
            data={"echoed": args.message},
            latency_ms=1.0,
        )


class FailingTool(Tool):
    """A test tool that always fails."""
    name = "fail"
    description = "Always fails."
    args_model = EchoArgs

    async def execute(self, args: EchoArgs) -> ToolResult:
        return ToolResult(
            success=False,
            data={},
            error="intentional_failure",
            latency_ms=1.0,
        )


class ExplodingTool(Tool):
    """A test tool that raises an exception."""
    name = "explode"
    description = "Always raises."
    args_model = EchoArgs

    async def execute(self, args: EchoArgs) -> ToolResult:
        raise RuntimeError("kaboom!")


# =============================================================================
# Tests
# =============================================================================

class TestToolDispatcher:
    """Tests for the tool dispatcher."""

    def test_init_with_tools(self):
        """Dispatcher registers tools by name."""
        tools = [EchoTool()]
        dispatcher = ToolDispatcher(tools)
        assert "echo" in dispatcher.registry
        assert len(dispatcher.registry) == 1

    def test_init_rejects_duplicate_names(self):
        """Duplicate tool names raise ValueError."""
        with pytest.raises(ValueError, match="Duplicate tool name"):
            ToolDispatcher([EchoTool(), EchoTool()])

    def test_get_tool_schemas(self):
        """get_tool_schemas returns schema for each tool."""
        dispatcher = ToolDispatcher([EchoTool()])
        schemas = dispatcher.get_tool_schemas()
        assert len(schemas) == 1
        assert schemas[0]["name"] == "echo"

    async def test_dispatch_valid_call(self):
        """Valid call returns success result."""
        dispatcher = ToolDispatcher([EchoTool()])
        call = ToolCall(
            name="echo",
            arguments={"message": "hello"},
            reasoning="testing",
        )
        result = await dispatcher.dispatch(call)

        assert result.success is True
        assert result.data["echoed"] == "hello"

    async def test_dispatch_unknown_tool(self):
        """Unknown tool returns error, doesn't raise."""
        dispatcher = ToolDispatcher([EchoTool()])
        call = ToolCall(
            name="nonexistent",
            arguments={},
            reasoning="testing",
        )
        result = await dispatcher.dispatch(call)

        assert result.success is False
        assert result.error is not None
        assert "unknown_tool" in result.error

    async def test_dispatch_invalid_args(self):
        """Invalid arguments return error, don't call execute."""
        dispatcher = ToolDispatcher([EchoTool()])
        call = ToolCall(
            name="echo",
            arguments={},  # missing required "message"
            reasoning="testing",
        )
        result = await dispatcher.dispatch(call)

        assert result.success is False
        assert result.error is not None
        assert "invalid_arguments" in result.error

    async def test_dispatch_tool_failure(self):
        """Tool returning failure is passed through."""
        dispatcher = ToolDispatcher([FailingTool()])
        call = ToolCall(
            name="fail",
            arguments={"message": "test"},
            reasoning="testing",
        )
        result = await dispatcher.dispatch(call)

        assert result.success is False
        assert result.error == "intentional_failure"

    async def test_dispatch_tool_exception_caught(self):
        """Tool raising exception is caught, returns error."""
        dispatcher = ToolDispatcher([ExplodingTool()])
        call = ToolCall(
            name="explode",
            arguments={"message": "test"},
            reasoning="testing",
        )
        result = await dispatcher.dispatch(call)

        # Dispatcher catches the exception, returns error
        assert result.success is False
        assert result.error is not None
        assert "tool_exception" in result.error
        assert "kaboom" in result.error
