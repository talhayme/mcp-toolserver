"""A small, well-tested MCP server exposing internal tools to an LLM."""

from .server import TOOL_SCHEMAS, dispatch
from .tools import ToolError, calculate, date_difference, fetch_document, search_documents

__version__ = "0.1.0"

__all__ = [
    "TOOL_SCHEMAS", "dispatch", "ToolError",
    "search_documents", "fetch_document", "calculate", "date_difference",
]
