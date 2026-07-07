"""Reusable MCP (Model Context Protocol) tool loader.

Add an MCP server once in ``tools/mcp_servers.json``, then any agent pulls its
tools in one line:

    from tools.mcp import get_mcp_tools
    tools = get_mcp_tools(["travel"])          # or get_mcp_tools() for all servers
    agent = create_agent(model, tools=tools, ...)

Server config format (``mcp_servers.json``) — keyed by a short name:

    {
      "travel":  {"transport": "streamable_http", "url": "https://mcp.kiwi.com"},
      "github":  {"transport": "stdio", "command": "npx",
                  "args": ["-y", "@modelcontextprotocol/server-github"]}
    }

Values are passed straight to ``MultiServerMCPClient``, so any transport it
supports works. A retry interceptor (lifted from the original wedding planner)
makes flaky servers degrade to an error message instead of crashing the agent.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient
from mcp.shared.exceptions import McpError
from mcp.types import CallToolResult, TextContent

_CONFIG_PATH = Path(__file__).parent / "mcp_servers.json"

# MCP internal-error code — worth retrying; other codes are returned as-is.
RETRYABLE_MCP_CODES = {-32603}


class RetryMCPInterceptor:
    """Retries retryable MCP failures with backoff; never raises into the agent."""

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries

    async def __call__(self, request, handler):
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return await handler(request)
            except McpError as exc:
                last_error = exc
                if exc.error.code not in RETRYABLE_MCP_CODES:
                    return CallToolResult(
                        content=[TextContent(type="text", text=f"Tool call failed (non-retryable): {exc}")],
                        isError=False,
                    )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
            if attempt < self.max_retries - 1:
                await asyncio.sleep(2**attempt)

        return CallToolResult(
            content=[TextContent(type="text", text=f"Tool call failed after {self.max_retries} attempts: {last_error}")],
            isError=False,
        )


def load_server_config() -> dict:
    """Read the configured MCP servers (empty dict if the file is missing)."""
    if _CONFIG_PATH.exists():
        return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    return {}


async def aget_mcp_tools(server_names: list[str] | None = None) -> list:
    """Async: connect to the named servers (or all) and return their tools."""
    config = load_server_config()
    if server_names is not None:
        config = {k: v for k, v in config.items() if k in server_names}
    if not config:
        return []
    client = MultiServerMCPClient(config, tool_interceptors=[RetryMCPInterceptor()])
    return await client.get_tools()


def get_mcp_tools(server_names: list[str] | None = None) -> list:
    """Sync wrapper for module-level agent setup. Returns [] on failure so a
    down MCP server never blocks the agent from loading."""
    try:
        return asyncio.run(aget_mcp_tools(server_names))
    except RuntimeError as exc:
        # Called from inside a running event loop — use aget_mcp_tools instead.
        raise RuntimeError(
            "get_mcp_tools() can't run inside an event loop; await aget_mcp_tools()."
        ) from exc
    except Exception as exc:  # noqa: BLE001
        print(f"[mcp] failed to load tools for {server_names or 'all servers'}: {exc}")
        return []
