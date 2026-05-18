import asyncio
import logging
import os
import sys

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

# Ensure we can import tools.py whether run from project root or from mcp_server/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools import run_python, search_documents, validate_yaml

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,  # IMPORTANT: stdio is used for MCP - logs must go to stderr
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "msg": "%(message)s"}',
)
logger = logging.getLogger(__name__)

app = Server("kubernetes-assistant-tools")


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search_documents",
            description=(
                "Semantic search over the Kubernetes documentation corpus "
                "(official K8s concept pages). Use this to look up Kubernetes "
                "concepts, commands, resource definitions, or how-to information."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of top results to return (1-10).",
                        "default": 3,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="run_python",
            description=(
                "Execute Python code in a sandboxed subprocess. Captures stdout, "
                "stderr, and exit code. Code times out after 10 seconds. "
                "Useful for calculations, data processing, parsing, or "
                "demonstrating logic with concrete output."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute.",
                    },
                },
                "required": ["code"],
            },
        ),
        Tool(
            name="validate_yaml",
            description=(
                "Parse YAML and check that each document has the basic "
                "structure of a Kubernetes resource (apiVersion, kind, "
                "metadata.name). Returns parse errors or a per-document "
                "validation report."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "yaml_text": {
                        "type": "string",
                        "description": "YAML content to validate.",
                    },
                },
                "required": ["yaml_text"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    logger.info(f"Tool called: {name} with arg keys: {list(arguments.keys())}")

    try:
        if name == "search_documents":
            result = search_documents(
                query=arguments["query"],
                top_k=arguments.get("top_k", 3),
            )
        elif name == "run_python":
            result = run_python(code=arguments["code"])
        elif name == "validate_yaml":
            result = validate_yaml(yaml_text=arguments["yaml_text"])
        else:
            result = f"Unknown tool: {name}"

        return [TextContent(type="text", text=result)]

    except Exception as e:
        logger.error(f"Tool '{name}' failed: {e}")
        return [TextContent(type="text", text=f"Tool execution error: {e}")]


async def main():
    logger.info("Starting MCP server (stdio transport)...")
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
