"""Tool implementations, deliberately kept free of any MCP imports.

Separating the business logic from the protocol layer is what makes these
testable without standing up a server, and what lets the same functions be
exposed over HTTP or a queue later without a rewrite.

Every tool follows one rule: **invalid input raises ToolError with a message
written for the model**, not a stack trace. An LLM that receives "query must
not be empty" can correct itself; one that receives a traceback cannot.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional


class ToolError(Exception):
    """A user-correctable error. The message is shown to the model verbatim."""


# --------------------------------------------------------------------------
# A tiny in-memory corpus. A real deployment would point at a database or an
# internal API; the shape of the tool contract would not change.
# --------------------------------------------------------------------------

DOCUMENTS: List[Dict[str, Any]] = [
    {
        "id": "policy-refunds",
        "title": "Refund policy",
        "team": "support",
        "updated": "2026-02-11",
        "body": (
            "Customers may request a full refund within 14 days of purchase. "
            "After 14 days, refunds are issued as store credit only. "
            "Refunds are processed back to the original payment method and "
            "take 5 to 10 business days to appear."
        ),
    },
    {
        "id": "policy-pricing",
        "title": "Plan pricing",
        "team": "sales",
        "updated": "2026-03-02",
        "body": (
            "Starter costs 12 USD per month. Pro costs 29 USD per month and "
            "includes priority support and the audit log. Enterprise is "
            "custom-priced and requires an annual contract."
        ),
    },
    {
        "id": "runbook-deploy",
        "title": "Deployment runbook",
        "team": "engineering",
        "updated": "2026-01-24",
        "body": (
            "Deploys run from the main branch only. The pipeline runs unit "
            "tests, the evaluation gate and a smoke check against staging. "
            "A failed gate blocks the deploy. Rollback is a redeploy of the "
            "previous tag and takes about 4 minutes."
        ),
    },
    {
        "id": "runbook-incident",
        "title": "Incident response",
        "team": "engineering",
        "updated": "2026-04-08",
        "body": (
            "Page the on-call engineer through the escalation channel. "
            "Severity 1 means a full outage and requires a status page update "
            "within 15 minutes. Write the postmortem within 3 business days."
        ),
    },
    {
        "id": "onboarding-access",
        "title": "New hire access",
        "team": "people",
        "updated": "2026-05-19",
        "body": (
            "Accounts are provisioned on the first day through the identity "
            "provider. Production access requires manager approval and a "
            "completed security training module."
        ),
    },
]

MAX_QUERY_LENGTH = 200
MAX_SEARCH_LIMIT = 20


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).casefold()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text)).strip()


def _score(query: str, doc: Dict[str, Any]) -> float:
    """Term-overlap score weighted toward title matches."""
    terms = [t for t in _normalize(query).split() if len(t) > 1]
    if not terms:
        return 0.0

    title_tokens = set(_normalize(doc["title"]).split())
    body_tokens = set(_normalize(doc["body"]).split())

    hits = 0.0
    for term in terms:
        if term in title_tokens:
            hits += 1.0
        elif term in body_tokens:
            hits += 0.6
        elif any(token.startswith(term) for token in body_tokens | title_tokens):
            # Prefix credit so "deploy" still finds "deployment".
            hits += 0.3
    return hits / len(terms)


def search_documents(query: str, limit: int = 3, team: Optional[str] = None) -> Dict[str, Any]:
    """Search the knowledge base and return ranked excerpts.

    Returning an explicit empty result with a reason — rather than an error —
    lets the model tell the user "nothing matched" instead of retrying blindly.
    """
    if not isinstance(query, str) or not query.strip():
        raise ToolError("query must be a non-empty string")
    if len(query) > MAX_QUERY_LENGTH:
        raise ToolError(f"query must be at most {MAX_QUERY_LENGTH} characters, got {len(query)}")
    if not isinstance(limit, int) or isinstance(limit, bool):
        raise ToolError("limit must be an integer")
    if not 1 <= limit <= MAX_SEARCH_LIMIT:
        raise ToolError(f"limit must be between 1 and {MAX_SEARCH_LIMIT}, got {limit}")

    candidates = DOCUMENTS
    if team is not None:
        if not isinstance(team, str) or not team.strip():
            raise ToolError("team must be a non-empty string when provided")
        known_teams = sorted({d["team"] for d in DOCUMENTS})
        if team not in known_teams:
            raise ToolError(f"unknown team {team!r}; known teams are: {', '.join(known_teams)}")
        candidates = [d for d in DOCUMENTS if d["team"] == team]

    scored = [(doc, _score(query, doc)) for doc in candidates]
    matches = sorted(
        ((doc, s) for doc, s in scored if s > 0),
        key=lambda pair: (-pair[1], pair[0]["id"]),
    )[:limit]

    return {
        "query": query,
        "count": len(matches),
        "results": [
            {
                "id": doc["id"],
                "title": doc["title"],
                "team": doc["team"],
                "updated": doc["updated"],
                "relevance": round(score, 3),
                "excerpt": doc["body"][:200],
            }
            for doc, score in matches
        ],
        # Grounding instruction travels with the data so the model cannot
        # separate the two.
        "note": (
            "No documents matched. Say so rather than answering from memory."
            if not matches
            else "Answer only from these excerpts. Cite the document id."
        ),
    }


def fetch_document(document_id: str) -> Dict[str, Any]:
    """Return one full document by id."""
    if not isinstance(document_id, str) or not document_id.strip():
        raise ToolError("document_id must be a non-empty string")

    for doc in DOCUMENTS:
        if doc["id"] == document_id:
            return dict(doc)

    available = ", ".join(d["id"] for d in DOCUMENTS)
    raise ToolError(f"no document with id {document_id!r}. Available ids: {available}")


# --------------------------------------------------------------------------
# Deterministic utility tools. These exist because they are the things LLMs
# reliably get wrong on their own: arithmetic and date maths.
# --------------------------------------------------------------------------

_ALLOWED_MATH_NAMES = {
    "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
    "pow": pow, "sqrt": math.sqrt, "floor": math.floor, "ceil": math.ceil,
    "log": math.log, "log10": math.log10, "exp": math.exp,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "pi": math.pi, "e": math.e,
}

_MATH_PATTERN = re.compile(r"^[0-9a-zA-Z_+\-*/%.,()\s]+$")


def calculate(expression: str) -> Dict[str, Any]:
    """Evaluate an arithmetic expression safely.

    `eval` is the obvious implementation and the wrong one: it hands arbitrary
    code execution to whatever text the model produced. This restricts the
    character set, compiles the expression, and rejects any name that is not
    an explicitly allowed maths function.
    """
    if not isinstance(expression, str) or not expression.strip():
        raise ToolError("expression must be a non-empty string")
    if len(expression) > 200:
        raise ToolError("expression is too long (limit 200 characters)")
    if not _MATH_PATTERN.match(expression):
        raise ToolError(
            "expression contains unsupported characters; only numbers, "
            "operators, parentheses and maths function names are allowed"
        )
    if "__" in expression:
        raise ToolError("expression contains a forbidden token")

    try:
        code = compile(expression, "<calculate>", "eval")
    except SyntaxError as exc:
        raise ToolError(f"expression is not valid: {exc.msg}") from exc

    for name in code.co_names:
        if name not in _ALLOWED_MATH_NAMES:
            allowed = ", ".join(sorted(_ALLOWED_MATH_NAMES))
            raise ToolError(f"name {name!r} is not allowed. Allowed names: {allowed}")

    try:
        value = eval(code, {"__builtins__": {}}, dict(_ALLOWED_MATH_NAMES))  # noqa: S307
    except ZeroDivisionError as exc:
        raise ToolError("division by zero") from exc
    except (ValueError, OverflowError) as exc:
        raise ToolError(f"could not evaluate: {exc}") from exc

    if isinstance(value, complex):
        raise ToolError("expression produced a complex number, which is not supported")
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        raise ToolError("expression produced a non-finite result")

    return {"expression": expression, "result": value}


def date_difference(start: str, end: str) -> Dict[str, Any]:
    """Days between two ISO-8601 dates (YYYY-MM-DD)."""
    def _parse(value: str, label: str) -> datetime:
        if not isinstance(value, str) or not value.strip():
            raise ToolError(f"{label} must be a non-empty ISO-8601 date string")
        try:
            return datetime.strptime(value.strip(), "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError as exc:
            raise ToolError(f"{label} {value!r} is not a valid YYYY-MM-DD date") from exc

    start_dt = _parse(start, "start")
    end_dt = _parse(end, "end")
    delta: timedelta = end_dt - start_dt

    return {
        "start": start_dt.date().isoformat(),
        "end": end_dt.date().isoformat(),
        "days": delta.days,
        "direction": "forward" if delta.days >= 0 else "backward",
    }
