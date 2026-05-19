"""
Prompts for the ReAct agent.

build_react_prompt(tools) takes a list of MCP-discovered Tool objects and
generates the system prompt dynamically. This way, if a new tool is added to
the MCP server, the agent picks it up automatically via session.list_tools()
without needing to update this file.
"""

from typing import Any


def _format_tool(tool: Any) -> str:
    """
    Format a single tool's description for the prompt.

    The MCP Tool object has .name, .description, and .inputSchema.
    inputSchema is a JSON schema dict.
    """
    name = getattr(tool, "name", "unknown")
    description = getattr(tool, "description", "")
    schema = getattr(tool, "inputSchema", {}) or {}
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    # Build a concise arg list: name(type, required/optional)
    arg_strs = []
    for arg_name, arg_spec in properties.items():
        arg_type = arg_spec.get("type", "any")
        is_required = arg_name in required
        marker = "" if is_required else "?"
        arg_strs.append(f"{arg_name}{marker}: {arg_type}")

    args_signature = ", ".join(arg_strs)
    return f"- {name}({args_signature}): {description.strip()}"


def build_react_prompt(tools: list[Any]) -> str:
    """
    Build the ReAct system prompt from a list of discovered MCP tools.

    Args:
        tools: list of MCP Tool objects (from session.list_tools().tools)

    Returns:
        a fully-formatted system prompt string ready to send to the model.
    """
    if tools:
        tool_descriptions = "\n".join(_format_tool(t) for t in tools)
        tool_names = ", ".join(t.name for t in tools)
    else:
        # Fallback: static list (used in test contexts where MCP isn't connected)
        tool_descriptions = (
            "- search_documents(query: string, top_k?: integer): "
            "Search the Kubernetes documentation corpus for relevant passages.\n"
            "- run_python(code: string): "
            "Execute a Python snippet in a sandboxed subprocess.\n"
            "- validate_yaml(yaml_text: string): "
            "Validate a Kubernetes YAML manifest for syntax and basic correctness."
        )
        tool_names = "search_documents, run_python, validate_yaml"

    return f"""You are a Kubernetes expert assistant. You answer questions by reasoning step-by-step and, when needed, using tools.

You have access to these tools (discovered dynamically from the MCP server):
{tool_descriptions}

Use the ReAct format. For each step output EXACTLY:

Thought: <your reasoning>
Action: <tool_name>
Action Input: <JSON object on a single line>

After you call an Action, the user will send you the Observation. Repeat the cycle until you have enough information.

When you have the final answer, output EXACTLY:

Final Answer: <answer>

IMPORTANT RULES:
1. The Action MUST be one of: {tool_names}
2. The Action Input MUST be a valid JSON object on a single line.
3. Do NOT fabricate Observations - wait for the real one.
4. Do NOT call the same tool with the same arguments twice in a row.
5. If you already have enough information, give a Final Answer immediately - don't loop.
6. For simple K8s knowledge questions, give a Final Answer directly without calling tools.
7. For YAML validation, ALWAYS use validate_yaml. For math/calculations, use run_python.

EXAMPLES:

Question: What is a Kubernetes Pod?
Thought: This is a knowledge question I can answer directly without tools.
Final Answer: A Pod is the smallest deployable unit in Kubernetes...

Question: Validate this manifest: apiVersion: v1\\nkind: Pod\\nmetadata: name: test
Thought: The user wants YAML validation - I'll use validate_yaml.
Action: validate_yaml
Action Input: {{"yaml_text": "apiVersion: v1\\nkind: Pod\\nmetadata:\\n  name: test"}}
(Observation will follow)
Thought: The validator returned the result. I'll give the final answer.
Final Answer: The manifest is valid. <explanation>

## Now respond to the user's question.
"""


# Backwards-compatible static prompt for tests that import REACT_PROMPT directly.
# This is the fallback prompt with hardcoded tools.
REACT_PROMPT = build_react_prompt([])
