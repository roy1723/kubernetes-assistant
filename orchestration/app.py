import asyncio
import logging
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "agent"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
import httpx
from router import Router

from agent import ReActAgent

logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "msg": "%(message)s"}',
)
logger = logging.getLogger(__name__)

import os

INFERENCE_URL = os.getenv("INFERENCE_URL", "http://localhost:8000")

# ---------- Lifecycle ----------

_agent: ReActAgent | None = None
_router: Router | None = None
_init_lock = asyncio.Lock()
_run_lock = asyncio.Lock()


async def get_router() -> Router:
    global _router
    if _router is None:
        async with _init_lock:
            if _router is None:
                logger.info("Initializing router...")
                _router = Router()
    return _router


async def get_agent() -> ReActAgent:
    global _agent
    if _agent is None:
        async with _init_lock:
            if _agent is None:
                logger.info("Initializing ReAct agent (launching MCP server)...")
                a = ReActAgent()
                await a.start()
                _agent = a
                logger.info("Agent ready.")
    return _agent


# ---------- Direct inference (no agent, no tools) ----------

DIRECT_SYSTEM_PROMPT = (
    "You are a Kubernetes expert assistant. Answer the user's question "
    "concisely and accurately, with concrete kubectl commands and YAML "
    "examples where appropriate. Keep the answer focused and avoid "
    "rambling. If you don't know something, say so."
)


async def call_inference_direct(query: str) -> str:
    """Call the FastAPI inference server directly. No tools, no agent."""
    payload = {
        "messages": [
            {"role": "system", "content": DIRECT_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        "temperature": 0.4,
        "max_tokens": 512,
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(f"{INFERENCE_URL}/chat", json=payload)
        r.raise_for_status()
    return r.json()["response"].strip()


# ---------- Casual ----------

CASUAL_RESPONSE = (
    "Hi! I'm a Kubernetes assistant. I can help with:\n\n"
    "- **Concepts and commands**: \"how do I scale a deployment?\", "
    "\"what's the difference between a ConfigMap and a Secret?\"\n"
    "- **YAML validation**: paste a manifest and I'll check it\n"
    "- **Calculations**: \"how many pods fit on 5 nodes with 16GB each?\"\n\n"
    "What would you like to know?"
)


# ---------- Trace formatting (for agent path) ----------

def format_trace_summary(trace: list) -> str:
    tool_calls = [t for t in trace if t.get("type") == "tool_call"]
    if not tool_calls:
        return ""

    parts = []
    for tc in tool_calls:
        tool = tc.get("tool", "?")
        args = tc.get("input", {})
        if isinstance(args, dict) and args:
            first_value = next(iter(args.values()))
            preview = str(first_value)[:60].replace("\n", " ")
            if len(str(first_value)) > 60:
                preview += "..."
            parts.append(f"`{tool}({preview!r})`")
        else:
            parts.append(f"`{tool}()`")

    return f"*Tools used: {' -> '.join(parts)}*\n\n"


# ---------- Main chat handler ----------

async def respond(message: str, history: list) -> str:
    if not message or not message.strip():
        return "Please ask a Kubernetes question."

    # Step 1: Classify
    try:
        router = await get_router()
        classification = await router.classify(message)
    except Exception as e:
        logger.error(f"Router failed: {e}. Defaulting to direct path.")
        classification = "direct"

    # Step 2: Route
    route_label = f"*Route: {classification}*\n\n"

    if classification == "casual":
        return route_label + CASUAL_RESPONSE

    elif classification == "direct":
        try:
            answer = await call_inference_direct(message)
            return route_label + answer
        except Exception as e:
            logger.error(f"Direct inference failed: {e}")
            return route_label + f"**Error calling inference server:** {e}"

    elif classification == "tools":
        try:
            agent = await get_agent()
        except Exception as e:
            logger.error(f"Agent init failed: {e}")
            return (
                route_label
                + f"**Failed to initialize agent:** {e}\n\n"
                "Make sure Ollama and the FastAPI server are running."
            )

        async with _run_lock:
            try:
                answer, trace = await agent.run(message)
            except Exception as e:
                logger.error(f"agent.run failed: {e}")
                return route_label + f"**Agent error:** {e}"

        header = format_trace_summary(trace)
        return route_label + header + answer

    else:
        return route_label + "Could not classify query. Please rephrase."


# ---------- UI ----------

EXAMPLES = [
    # Casual
    "Hi! What can you do?",
    # Direct
    "What's the difference between a ConfigMap and a Secret?",
    # Tools
    "Is this YAML valid?\napiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: app-config\ndata:\n  key1: value1",
    # Tools
    "Calculate how many pods fit on 3 nodes with 32GB each, if each pod needs 4GB.",
]

DESCRIPTION = """
A fine-tuned Phi-3-mini Kubernetes assistant with three response paths:

- **casual** -- greetings and off-topic queries get a static response
- **direct** -- simple K8s questions go straight to the fine-tuned model
- **tools** -- queries needing validation or computation use the ReAct agent with MCP tools

The router classifies your query first, then dispatches to the right handler.
"""


demo = gr.ChatInterface(
    fn=respond,
    title="Kubernetes Assistant",
    description=DESCRIPTION,
    examples=EXAMPLES,
)


if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),
    )
