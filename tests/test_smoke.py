import importlib
import os
import sys

# Make project root importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "agent"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "orchestration"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "mcp_server"))


def test_agent_imports():
    """Agent module imports without errors."""
    m = importlib.import_module("agent")
    assert hasattr(m, "ReActAgent")
    assert hasattr(m, "parse_output")
    assert hasattr(m, "normalize_tool_name")


def test_router_imports():
    """Router module imports without errors."""
    m = importlib.import_module("router")
    assert hasattr(m, "Router")
    assert hasattr(m, "VALID_LABELS")
    assert m.VALID_LABELS == {"casual", "direct", "tools"}


def test_prompts_imports():
    """Prompts module imports and provides REACT_PROMPT."""
    m = importlib.import_module("prompts")
    assert hasattr(m, "REACT_PROMPT")
    assert len(m.REACT_PROMPT) > 500  # non-trivial prompt content


def test_tool_name_aliasing():
    """Hallucinated tool names map to canonical names."""
    from agent import normalize_tool_name

    assert normalize_tool_name("search_documentation") == "search_documents"
    assert normalize_tool_name("validate") == "validate_yaml"
    assert normalize_tool_name("python") == "run_python"
    assert normalize_tool_name("search_documents") == "search_documents"


def test_router_keyword_routes():
    """Keyword router classifies obvious cases correctly without an LLM."""
    from router import Router

    r = Router()

    # Casual
    assert r.keyword_route("hi") == "casual"
    assert r.keyword_route("thanks!") == "casual"
    assert r.keyword_route("what can you do for me") == "casual"

    # Tools (YAML block)
    assert r.keyword_route("apiVersion: v1\nkind: Pod") == "tools"

    # Tools (validation request)
    assert r.keyword_route("validate this manifest please") == "tools"

    # Tools (math + units)
    assert (
        r.keyword_route("how many pods fit on 4 nodes with 16GB each") == "tools"
    )

    # Ambiguous -> None (falls through to LLM)
    assert r.keyword_route("what is a Service") is None


def test_parse_output_final_answer():
    """Parser extracts Final Answer correctly."""
    from agent import parse_output

    text = "Thought: I know this.\nFinal Answer: Use kubectl get pods."
    parsed = parse_output(text)
    assert parsed["type"] == "final"
    assert "kubectl get pods" in parsed["answer"]


def test_parse_output_action():
    """Parser extracts Action + Action Input correctly."""
    from agent import parse_output

    text = (
        'Thought: I need to validate.\n'
        'Action: validate_yaml\n'
        'Action Input: {"yaml_text": "apiVersion: v1"}'
    )
    parsed = parse_output(text)
    assert parsed["type"] == "action"
    assert parsed["tool"] == "validate_yaml"
    assert parsed["input"] == {"yaml_text": "apiVersion: v1"}
