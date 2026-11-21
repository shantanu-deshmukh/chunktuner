"""FastMCP entrypoint: ``chunk-tune-mcp`` (stdio)."""

import logging
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP


def create_mcp_app() -> "FastMCP":
    """Build the FastMCP app (tools, resources, prompts). Used by stdio ``run()`` and tests."""
    from mcp.server.fastmcp import FastMCP

    from chunktuner.mcp.prompts import register_prompts
    from chunktuner.mcp.resources import register_resources
    from chunktuner.mcp.tools import register_tools

    mcp = FastMCP("chunktuner")
    register_tools(mcp)
    register_resources(mcp)
    register_prompts(mcp)
    return mcp


def run() -> None:
    """Run the MCP server on stdio. Never write to stdout (JSON-RPC channel)."""
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(levelname)s %(name)s %(message)s",
    )
    create_mcp_app().run()


if __name__ == "__main__":
    run()
