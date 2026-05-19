"""
orchestration/app.py

Gradio chat UI for the Kubernetes Assistant.

The Router classifies each query as casual / direct / tools, then dispatches
to the right handler:
  - casual: static greeting
  - direct: FastAPI inference server (fine-tuned phi3-kubernetes)
  - tools:  ReAct agent (which uses MCP tools via stdio)

Session memory: Gradio gr.State holds per-conversation history.
"""

import asyncio
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "agent"),
)

import gradio as gr
import httpx
from router import Router

from agent import ReActAgent

# Human-readable console log format (force=True overrides any prior config)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
logger = logging.getLogger(__name__)

# Suppress noisy library logs - we log our own HTTP calls
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)

INFERENCE_URL = os.getenv("INFERENCE_URL", "http://localhost:8000")

logger.info("=" * 78)
logger.info("Orchestration (Gradio) starting")
logger.info(f"  INFERENCE_URL = {INFERENCE_URL}")
logger.info("=" * 78)


DIRECT_SYSTEM_PROMPT = (
    "You are a Kubernetes expert assistant. Answer the user's question "
    "concisely and accurately, with concrete kubectl commands and YAML "
    "examples where appropriate."
)

CASUAL_RESPONSE = (
    "Hi! I'm a Kubernetes assistant. I can help with:\n\n"
    "- **Concepts and commands**: 'how do I scale a deployment?', "
    "'what's the difference between a ConfigMap and a Secret?'\n"
    "- **YAML validation**: paste a manifest and I'll check it\n"
    "- **Calculations**: 'how many pods fit on 5 nodes with 16GB each?'\n\n"
    "What would you like to know?"
)

MAX_HISTORY = 6


# ---------- Lifecycle ----------

agent: ReActAgent | None = None
router: Router | None = None
init_lock = asyncio.Lock()


async def init_services():
    """Initialize agent and router (idempotent - safe to call multiple times)."""
    global agent, router
    async with init_lock:
        if agent is None:
            logger.info(">>> Initializing ReAct agent (starting MCP subprocess)...")
            agent_start = time.time()
            agent = ReActAgent(inference_url=INFERENCE_URL)
            await agent.start()
            agent_ms = int((time.time() - agent_start) * 1000)
            logger.info(
                f"<<< Agent ready in {agent_ms}ms  "
                f"tools={sorted(agent.discovered_tool_names)}"
            )
        if router is None:
            logger.info(">>> Initializing router...")
            router = Router(inference_url=INFERENCE_URL)
            logger.info("<<< Router ready (hybrid keyword + LLM classifier)")


async def eager_init():
    """Called at app startup so first user query doesn't pay init cost."""
    logger.info("Pre-warming services so first query is fast...")
    await init_services()
    logger.info(">>> ALL SERVICES READY - waiting for user queries <<<")
    logger.info("")


# ---------- Direct path ----------

async def call_inference_direct(
    query: str, history: list[dict] | None = None
) -> str:
    """Call FastAPI /chat directly with system prompt + history + query."""
    messages: list[dict] = [{"role": "system", "content": DIRECT_SYSTEM_PROMPT}]
    if history:
        messages.extend(history[-MAX_HISTORY:])
    messages.append({"role": "user", "content": query})

    logger.info(f"    --> sending {len(messages)} messages to FastAPI /chat")

    payload = {
        "messages": messages,
        "temperature": 0.4,
        "max_tokens": 1024,
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(f"{INFERENCE_URL}/chat", json=payload)
        r.raise_for_status()
    return r.json()["response"].strip()


# ---------- Main handler ----------

async def respond(message: str, chat_history: list, session_state: dict):
    """Async Gradio handler — one turn of the chat."""
    turn_start = time.time()
    await init_services()
    assert router is not None
    assert agent is not None

    # Ensure session state has the history key
    if "history" not in session_state:
        session_state["history"] = []
    history = session_state["history"]

    logger.info("")
    logger.info("=" * 78)
    logger.info(f"NEW USER MESSAGE  (history: {len(history) // 2} prior turns)")
    logger.info(f"  > {message[:200]}{'...' if len(message) > 200 else ''}")
    logger.info("=" * 78)

    # ---- Route ----
    route_start = time.time()
    route = await router.classify(message)
    route_ms = int((time.time() - route_start) * 1000)
    logger.info(f"[ROUTE]  decision={route.upper()}  ({route_ms} ms)")

    # ---- Dispatch ----
    handler_start = time.time()
    tool_calls_used: list[str] = []

    if route == "casual":
        logger.info("[HANDLER] casual route - returning static response")
        answer = CASUAL_RESPONSE
    elif route == "direct":
        logger.info("[HANDLER] direct route - calling fine-tuned model via FastAPI")
        try:
            answer = await call_inference_direct(message, history=history)
        except Exception as e:
            logger.error(f"[HANDLER] direct FAILED: {e}")
            answer = f"**Error calling inference server:** {e}"
    elif route == "tools":
        logger.info("[HANDLER] tools route - invoking ReAct agent")
        try:
            answer, trace = await agent.run(message, history=history)
            # Log each tool call inline as agent progresses
            for step in trace:
                if step.get("type") == "tool_call":
                    tool_name = step.get("tool", "?")
                    tool_calls_used.append(tool_name)
                    args_str = str(step.get("args", {}))[:100]
                    logger.info(f"    [tool] {tool_name}  args={args_str}")
                elif step.get("type") == "tool_result":
                    result_str = str(step.get("result", ""))[:120]
                    logger.info(f"    [result] {result_str}")
            if not tool_calls_used:
                logger.info("    [agent] no tools invoked - went straight to answer")
        except Exception as e:
            logger.error(f"[HANDLER] agent FAILED: {e}")
            answer = f"**Agent error:** {e}"
    else:
        answer = f"_Unknown route: {route}_"

    handler_ms = int((time.time() - handler_start) * 1000)
    total_ms = int((time.time() - turn_start) * 1000)

    summary = f"route={route_ms}ms  handler={handler_ms}ms  total={total_ms}ms"
    if tool_calls_used:
        summary += f"  tools={tool_calls_used}"
    logger.info(f"[DONE]  {summary}  answer_chars={len(answer)}")
    logger.info(f"  < {answer[:150]}{'...' if len(answer) > 150 else ''}")
    logger.info("=" * 78)
    logger.info("")

    # Update session history
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})
    session_state["history"] = history[-MAX_HISTORY:]

    # Gradio 6 messages format
    chat_history = chat_history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": f"_Route: {route}_\n\n{answer}"},
    ]

    return "", chat_history, session_state


# ---------- UI ----------

with gr.Blocks(title="Kubernetes Assistant") as demo:
    gr.Markdown("# Kubernetes Assistant")
    gr.Markdown(
        "A fine-tuned Phi-3-mini Kubernetes assistant with three response paths:\n\n"
        "- **casual** -- greetings and off-topic queries get a static response\n"
        "- **direct** -- simple K8s questions go straight to the fine-tuned model\n"
        "- **tools** -- queries needing validation or computation use the ReAct "
        "agent with MCP tools\n\n"
        "The router classifies your query first, then dispatches to the right "
        "handler. Conversation memory is preserved within this chat session."
    )

    chatbot = gr.Chatbot(label="Chatbot", height=500)
    msg = gr.Textbox(
        label="Your message",
        placeholder="Ask me about Kubernetes...",
    )
    clear = gr.Button("Clear conversation")

    session_state = gr.State({"history": []})

    msg.submit(
        respond,
        inputs=[msg, chatbot, session_state],
        outputs=[msg, chatbot, session_state],
    )

    def clear_session():
        logger.info("[UI] user cleared conversation; session state reset")
        return [], {"history": []}

    clear.click(clear_session, outputs=[chatbot, session_state])

    # Pre-warm agent and router so first user query is fast
    demo.load(eager_init)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),
    )
