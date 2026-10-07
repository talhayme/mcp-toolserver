# mcp-toolserver — an MCP server built the way a production one should be

A Model Context Protocol server exposing four tools to an LLM: a grounded
document search, a document fetch, a sandboxed calculator and date
arithmetic.

The tools themselves are small on purpose. What this repository actually
demonstrates is the part that decides whether tool calling works in
production: **strict schemas, errors written for the model, and a protocol
layer thin enough that the logic underneath stays testable.**

```bash
git clone https://github.com/talhayme/mcp-toolserver
cd mcp-toolserver
pip install -e ".[dev]"
pytest -q          # 50 tests, no SDK and no API key required
```

## Connect it to Claude Desktop

```bash
pip install -e ".[mcp]"
```

Then add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "toolserver": {
      "command": "python",
      "args": ["-m", "mcp_toolserver.server"]
    }
  }
}
```

Restart Claude Desktop and ask it something like *"What's our refund policy
after 20 days, and what would 14 months of Pro cost with a 15% discount?"* —
it will call `search_documents` and then `calculate`.

## The tools

| Tool | Purpose |
|---|---|
| `search_documents` | Ranked excerpts from the knowledge base, with a grounding instruction attached |
| `fetch_document` | Full text of one document by id |
| `calculate` | Arithmetic evaluated exactly, in a sandbox |
| `date_difference` | Days between two ISO dates |

The last two exist because they are precisely what language models get wrong
unaided: arithmetic and date maths.

## Three decisions worth explaining

### 1. Errors are written for the model, not for a log

Every failure path raises `ToolError` with a message the model can act on:

```python
search_documents("refund", team="marketing")
# ToolError: unknown team 'marketing'; known teams are: engineering, people, sales, support

dispatch("calculate", {"expr": "2+2"})
# ToolError: invalid arguments for 'calculate': calculate() got an unexpected keyword argument 'expr'
```

A model that receives *"known teams are: engineering, people, sales,
support"* fixes its next call. A model that receives a stack trace retries
the same mistake, or gives up and invents an answer. The protocol layer
returns the message as tool output rather than raising, so the conversation
continues instead of failing.

### 2. The calculator is a sandbox, not an `eval`

`eval()` on a model-generated string is remote code execution with extra
steps. This implementation applies four separate restrictions:

```python
calculate("__import__('os').system('rm -rf /')")   # ToolError: forbidden token
calculate("().__class__.__bases__[0].__subclasses__()")  # ToolError: forbidden token
calculate("open('/etc/passwd').read()")            # ToolError: name 'open' is not allowed
calculate("1 / 0")                                 # ToolError: division by zero
```

A character allow-list, a `__` ban, `compile()` with every name checked
against an explicit maths allow-list, and `eval` with `__builtins__`
emptied. Five escape attempts are in the test suite as regression tests.

### 3. Schemas are strict, so the model gets it right first time

```json
{
  "limit": { "type": "integer", "minimum": 1, "maximum": 20, "default": 3 },
  "team":  { "type": "string", "enum": ["support", "sales", "engineering", "people"] },
  "additionalProperties": false
}
```

`additionalProperties: false` stops the model inventing parameters. Bounds
and enums turn a class of runtime errors into something the client catches
before the call is made. A test asserts that every tool has a description,
every property is documented, and every schema is strict — vague schemas
cause retry loops that cost tokens and latency.

## Grounding travels with the data

`search_documents` returns an instruction alongside its results:

```json
{
  "count": 0,
  "results": [],
  "note": "No documents matched. Say so rather than answering from memory."
}
```

An empty search is the moment a model is most likely to hallucinate. Putting
the instruction in the payload means it cannot drift out of sync with the
system prompt, and it is right where the model is looking.

## Architecture

```
mcp_toolserver/
  tools.py     # pure functions, zero MCP imports — all the logic lives here
  server.py    # schemas, dispatch, ToolError -> model-readable output. Thin.
```

`tools.py` imports nothing from the MCP SDK, which is why the test suite runs
without it and why the same functions could be served over HTTP tomorrow
without a rewrite. `build_server()` imports the SDK lazily and fails with an
actionable message when it is missing:

```
RuntimeError: The MCP SDK is not installed. Install it with:
    pip install 'mcp-toolserver[mcp]'
The tool logic and its tests run without it.
```

## Tests

```bash
pytest tests/ -q --cov=mcp_toolserver --cov-report=term-missing
# 50 passed — 84% coverage
```

Covering: ranking and prefix matching, the empty-result grounding path,
every validation branch, five sandbox-escape attempts, leap-year date
handling, dispatch with wrong and missing arguments, and a schema
completeness check.

CI runs the suite on Python 3.10/3.11/3.12, then installs the real MCP SDK,
builds the server, asserts the registered tool names match the declared
schemas, and makes a live call through the protocol layer — including a
failing one, to confirm errors come back readable rather than as a crash.

That job earned its place: the first version of this server was written
against an older SDK decorator API and CI caught it on the first push.

## Why this exists

I build custom MCP servers connecting LLMs to internal tools and services as
part of my AI-implementation consulting work. This is a standalone, readable
extract of the patterns I use — not production code, and deliberately small
enough to read in one sitting.

## License

MIT
