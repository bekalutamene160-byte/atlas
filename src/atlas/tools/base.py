"""
Tool protocol — the contract every tool must follow.

Design principle: tools are the agent's "hands." They let it act on
the world (retrieve, search, compute, fetch). Every tool has:
- A name (unique identifier, used by the planner)
- A description (shown to the LLM so it knows when to pick this tool)
- An args_model (Pydantic schema for validated arguments)
- An execute() method (async, returns ToolResult)

Why this abstraction matters:
1. The dispatcher validates arguments BEFORE execution — invalid args
   never reach the tool. This prevents crashes from malformed LLM output.
2. Tools are interchangeable — adding a new tool doesn't change the
   dispatcher or the agent loop. Just register it.
3. Tools can be tested in isolation — pass args, check result. No need
   to spin up the whole agent.

Pro tip for advanced projects: at scale, add a "side_effects" flag to
each tool. Read-only tools (retrieve, search) are safe to retry.
Side-effecting tools (send_email, write_file) require idempotency keys
and human approval. ATLAS v1 has only read-only tools, so we skip this.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Type

from pydantic import BaseModel

from atlas.models import ToolArgs, ToolResult


class Tool(ABC):
    """Abstract base class for all tools.

    Subclasses must define:
    - name: str — unique identifier
    - description: str — shown to the LLM
    - args_model: Type[ToolArgs] — Pydantic schema for arguments
    - execute(args): ToolResult — async execution

    The dispatcher uses these to validate and route tool calls.
    """

    # Class attributes — subclasses MUST override these
    name: str = ""
    description: str = ""
    args_model: Type[ToolArgs]

    @abstractmethod
    async def execute(self, args: ToolArgs) -> ToolResult:
        """Execute the tool with validated arguments.

        Args:
            args: Pydantic-validated arguments (guaranteed to match args_model)

        Returns:
            ToolResult with success/failure, data, and latency
        """
        ...

    def get_schema(self) -> dict:
        """Return JSON schema for this tool.

        Used to tell the LLM what tools are available and what
        arguments they accept. The planner prompt includes this.
        """
        return {
            "name": self.name,
            "description": self.description,
            "args": self.args_model.model_json_schema(),
        }
