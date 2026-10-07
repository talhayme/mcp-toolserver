"""MCP protocol layer.

Thin by design: it declares schemas, dispatches to :mod:`mcp_toolserver.tools`
and converts :class:`ToolError` into a protocol-level error the model can act
on. All behaviour worth testing lives in ``tools.py``.

The JSON Schemas here are not decoration — they are how the model learns to
call the tool correctly on the first attempt. Vague schemas produce retry
loops that cost tokens and latency.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from .tools import (
    MAX_QUERY_LENGTH,
    MAX_SEARCH_LIMIT,
    ToolError,
    calculate,
    date_difference,
    fetch_document,
    search_documents,
)

SERVER_NAME = "toolserver"
SERVER_VERSION = "0.1.0"


TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "name": "search_documents",
        "description": (
            "Search the internal knowledge base and return ranked excerpts. "
            "Use this before answering any question about company policy, "
            "pricing or runbooks. Answer only from the excerpts returned; if "
            "the result is empty, say that the knowledge base has no answer."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language search query.",
                    "minLength": 1,
                    "maxLength": MAX_QUERY_LENGTH,
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum documents to return.",
                    "minimum": 1,
                    "maximum": MAX_SEARCH_LIMIT,
                    "default": 3,
                },
                "team": {
                    "type": "string",
                    "description": "Optional filter. One of: support, sales, engineering, people.",
                    "enum": ["support", "sales", "engineering", "people"],
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "fetch_document",
        "description": (
            "Fetch the full text of one document by id. Use after "
            "search_documents when an excerpt is not enough."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "string",
                    "description": "Document id, e.g. 'policy-refunds'.",
                    "minLength": 1,
                },
            },
            "required": ["document_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "calculate",
        "description": (
            "Evaluate an arithmetic expression exactly. Use this for any "
            "calculation instead of computing it yourself."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Arithmetic expression, e.g. '(29 * 12) * 0.85'.",
                    "minLength": 1,
                    "maxLength": 200,
                },
            },
            "required": ["expression"],
            "additionalProperties": False,
        },
    },
    {
        "name": "date_difference",
        "description": "Count the days between two ISO-8601 dates (YYYY-MM-DD).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "start": {"type": "string", "description": "Start date, YYYY-MM-DD."},
                "end": {"type": "string", "description": "End date, YYYY-MM-DD."},
            },
            "required": ["start", "end"],
            "additionalProperties": False,
        },
    },
]


HANDLERS = {
    "search_documents": search_documents,
    "fetch_document": fetch_document,
    "calculate": calculate,
    "date_difference": date_difference,
}


def dispatch(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Route a tool call to its handler.

    Returns a result dict on success. Raises :class:`ToolError` for anything
    the caller can fix — an unknown tool, a bad argument name, a bad value.
    """
    handler = HANDLERS.get(name)
    if handler is None:
        raise ToolError(f"unknown tool {name!r}. Available tools: {', '.join(sorted(HANDLERS))}")

    if not isinstance(arguments, dict):
        raise ToolError("arguments must be an object")

    try:
        return handler(**arguments)
    except TypeError as exc:
        # Wrong or missing keyword arguments: report it as a usable message
        # rather than letting a TypeError surface as a server crash.
        raise ToolError(f"invalid arguments for {name!r}: {exc}") from exc


def build_server():
    """Construct the MCP server. Imported lazily so tests need no SDK."""
    try:
        from mcp.server import Server
        from mcp.types import TextContent, Tool
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "The MCP SDK is not installed. Install it with:\n"
            "    pip install 'mcp-toolserver[mcp]'\n"
            "The tool logic and its tests run without it."
        ) from exc

    server = Server(SERVER_NAME)

    @server.list_tools()
    async def list_tools() -> List[Tool]:
        return [
            Tool(
                name=spec["name"],
                description=spec["description"],
                inputSchema=spec["inputSchema"],
            )
            for spec in TOOL_SCHEMAS
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
        try:
            result = dispatch(name, arguments or {})
        except ToolError as exc:
            # isError keeps the model in the loop: it sees a correctable
            # message instead of the conversation failing.
            return [TextContent(type="text", text=f"Error: {exc}")]
        return [TextContent(type="text", text=json.dumps(result, indent=2, ensure_ascii=False))]

    return server


async def run_stdio() -> None:  # pragma: no cover - requires a live MCP client
    """Serve over stdio, the transport Claude Desktop uses."""
    from mcp.server.stdio import stdio_server

    server = build_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main() -> None:  # pragma: no cover - process entry point
    import asyncio

    asyncio.run(run_stdio())


if __name__ == "__main__":  # pragma: no cover
    main()
