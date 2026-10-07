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


def _describe(tool_name: str) -> str:
    """Look up the hand-written description for a tool."""
    for spec in TOOL_SCHEMAS:
        if spec["name"] == tool_name:
            return spec["description"]
    raise KeyError(tool_name)  # pragma: no cover - guarded by a schema test


def build_server():
    """Construct the MCP server. The SDK is imported lazily so the tool
    logic and its tests stay runnable without it installed.

    Each tool is registered as a typed function: the SDK derives the JSON
    Schema from the annotations, and the hand-written descriptions in
    ``TOOL_SCHEMAS`` supply the prose the model reads when choosing a tool.
    """
    try:
        from mcp.server import MCPServer
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "The MCP SDK is not installed. Install it with:\n"
            "    pip install 'mcp-toolserver[mcp]'\n"
            "The tool logic and its tests run without it."
        ) from exc

    server = MCPServer(name=SERVER_NAME, version=SERVER_VERSION)

    def _result(name: str, **kwargs: Any) -> str:
        """Dispatch and serialize, turning ToolError into a model-readable
        message rather than letting it surface as a transport failure."""
        try:
            payload = dispatch(name, kwargs)
        except ToolError as exc:
            return f"Error: {exc}"
        return json.dumps(payload, indent=2, ensure_ascii=False)

    @server.tool(name="search_documents", description=_describe("search_documents"))
    def search_documents_tool(query: str, limit: int = 3, team: str = "") -> str:
        return _result(
            "search_documents",
            query=query,
            limit=limit,
            **({"team": team} if team else {}),
        )

    @server.tool(name="fetch_document", description=_describe("fetch_document"))
    def fetch_document_tool(document_id: str) -> str:
        return _result("fetch_document", document_id=document_id)

    @server.tool(name="calculate", description=_describe("calculate"))
    def calculate_tool(expression: str) -> str:
        return _result("calculate", expression=expression)

    @server.tool(name="date_difference", description=_describe("date_difference"))
    def date_difference_tool(start: str, end: str) -> str:
        return _result("date_difference", start=start, end=end)

    return server


def main() -> None:  # pragma: no cover - process entry point
    """Serve over stdio, the transport Claude Desktop uses."""
    build_server().run(transport="stdio")


if __name__ == "__main__":  # pragma: no cover
    main()
