"""
orchestration/app.py

Gradio chat UI for the Kubernetes Assistant.

Compatible with Gradio 6.x:
  - gr.Chatbot uses the new messages format (list of {role, content} dicts)
  - theme is passed to demo.launch(), not gr.Blocks()
  - Async handlers are passed directly to msg.submit (no asyncio.run wrapper)

The Router classifies each query as casual / direct / tools, then dispatches:
  - casual: static greeting
  - direct: FastAPI inference server (fine-tuned phi3-kubernetes)
  - tools:  ReAct agent (which uses MCP tools via stdio)

Session memory: Gradio gr.State holds per-conversation history. Both the
agent and the direct path use this history for multi-turn context.

Environment variables:
  INFERENCE_URL - FastAPI inference server URL (default: http://localhost:8000)
"""

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "agent"),
)

import gradio as gr
import httpx

from agent import ReActAgent
from router import Router

logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "msg": "%(message)s"}',
)
logger = logging.getLogger(__name__)

INFERENCE_URL = os.getenv("INFERENCE_URL", "http://localhost:8000")

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

MAX_HISTORY = 6  # 3 user + 3 assistant turns retained


# ---------- Lifecycle ----------

agent: ReActAgent | None = None
router: Router | None = None


async def init_services():
    """Initialize agent and router on first use."""
    global agent, router
    if agent is None:
        agent = ReActAgent(inference_url=INFERENCE_URL)
        await agent.start()
        logger.info("Agent started.")
    if router is None:
        router = Router(inference_url=INFERENCE_URL)
        logger.info("Router initialized.")


# ---------- Direct path ----------

async def call_inference_direct(
    query: str, history: list[dict] | None = None
) -> str:
    """
    Call the FastAPI inference server directly. No tools, no agent.
    History is the prior turns in this session (capped).
    """
    messages: list[dict] = [{"role": "system", "content": DIRECT_SYSTEM_PROMPT}]
    if history:
        messages.extend(history[-MAX_HISTORY:])
    messages.append({"role": "user", "content": query})

    payload = {
        "messages": messages,
        "temperature": 0.4,
        "max_tokens": 512,
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(f"{INFERENCE_URL}/chat", json=payload)
        r.raise_for_status()
    return r.json()["response"].strip()


# ---------- Main handler ----------

async def respond(message: str, chat_history: list, session_state: dict):
    """
    Async Gradio handler.

    Args:
        message: user's current message
        chat_history: Gradio's display history (list of {role, content} dicts)
        session_state: per-session dict containing 'history' (model messages)

    Returns:
        (cleared_input, updated chat_history, updated session_state)
    """
    await init_services()
    assert router is not None
    assert agent is not None

    if "history" not in session_state:
        session_state["history"] = []

    history = session_state["history"]

    # Route the query
    route = await router.classify(message)
    logger.info(f"Route for '{message[:60]}...': {route}")

    # Dispatch
    if route == "casual":
        answer = CASUAL_RESPONSE
    elif route == "direct":
        try:
            answer = await call_inference_direct(message, history=history)
        except Exception as e:
            logger.error(f"Direct inference failed: {e}")
            answer = f"**Error calling inference server:** {e}"
    elif route == "tools":
        try:
            answer, trace = await agent.run(message, history=history)
        except Exception as e:
            logger.error(f"Agent run failed: {e}")
            answer = f"**Agent error:** {e}"
    else:
        answer = f"_Unknown route: {route}_"

    # Update session history with this turn's exchange
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})
    session_state["history"] = history[-MAX_HISTORY:]

    # Gradio 6 expects messages format: list of {role, content} dicts
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
        "- **tools** -- queries needing validation or computation use the ReAct agent with MCP tools\n\n"
        "The router classifies your query first, then dispatches to the right handler. "
        "Conversation memory is preserved within this chat session."
    )

    # Gradio 6: type="messages" is default and not a valid parameter
    chatbot = gr.Chatbot(label="Chatbot", height=500)
    msg = gr.Textbox(
        label="Your message",
        placeholder="Ask me about Kubernetes...",
    )
    clear = gr.Button("Clear conversation")

    # Per-session state - resets when "Clear" is pressed
    session_state = gr.State({"history": []})

    # Async handler passed directly; Gradio 6 manages the event loop
    msg.submit(
        respond,
        inputs=[msg, chatbot, session_state],
        outputs=[msg, chatbot, session_state],
    )

    def clear_session():
        return [], {"history": []}

    clear.click(clear_session, outputs=[chatbot, session_state])


if __name__ == "__main__":
    # Gradio 6: theme moved here from gr.Blocks() constructor
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),
    )