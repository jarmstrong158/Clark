"""clark.mcp — MCP server exposing Clark over the standard Model
Context Protocol so any MCP-aware host (Claude Desktop, Cursor,
Continue, Zed, ...) can drive Clark in natural language using its
own model.

This is a thin shim. Every tool delegates to a localhost
`clark serve` over HTTP. Nothing here re-implements inference.

Run:  clark mcp                        (stdio transport)
Env:  CLARK_API_URL  (default http://127.0.0.1:8000)

History note: this code used to live in a separate clark-mcp repo
that also shipped a Hermes-3 + Ollama local LLM client and a QLoRA
fine-tune pipeline. That branch was retired when the operations
dashboard (clark ops) became the primary UX; what remains here is
the MCP-host integration story, ~200 lines of stateless wrapper.
"""
__all__ = ["server", "tools", "client"]
