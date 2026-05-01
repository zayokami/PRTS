"""Echo MCP server fixture for smoke_p4.py.

In-process Python implementation using the official ``mcp`` SDK's FastMCP server.
Spawned via stdio by the smoke test; exposes one tool ``echo(text) -> str``
that returns its input verbatim. Smaller surface than ``server-filesystem``,
no Node dependency.

Run as a script::

    python scripts/_fixtures/echo_mcp_server.py
"""

from __future__ import annotations

import sys

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("echo")


@mcp.tool()
def echo(text: str) -> str:
    """Return the input text unchanged."""
    return text


@mcp.tool()
def shout(text: str) -> str:
    """Return the input text uppercased."""
    return text.upper()


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    mcp.run()
