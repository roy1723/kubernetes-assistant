"""
agent.py - ReAct agent for the Kubernetes Assistant.

Features:
  - Tool discovery: queries MCP server with session.list_tools() at startup,
    builds the system prompt dynamically from the response.
  - Session memory: optional `history` parameter to run() preserves multi-turn
    context within a session.
  - Token aggregation: per-turn input/output token counts are summed across
    all inference calls and logged in the JSON record.
  - Loop detection: same (tool, args) twice -> nudge; three times -> forced exit
  - Tool name aliasing: corrects common misspellings before calling MCP
  - Parse errors -> retry message asking model to use valid JSON

Environment variables:
  INFERENCE_URL - FastAPI inference server URL (default: http://localhost:8000)
"""

import json
import logging
import os
import re
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts import build_react_prompt

# ---------- Configuration ----------
INFERENCE_URL = os.getenv("INFERENCE_URL", "http://localhost:8000")
PROJECT_ROOT = Path(__file__).parent.parent
MCP_SERVER_SCRIPT = str(PROJECT_ROOT / "mcp_server" / "server.py")
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

MAX_STEPS = 3
OBSERVATION_MAX_LEN = 1500
INFERENCE_TIMEOUT = 180.0
MAX_HISTORY_MESSAGES = 6  # session history cap (3 user + 3 assistant)

TOOL_NAME_ALIASES = {
    "search_documentation": "search_documents",
    "search_documentations": "search_documents",
    "search_docs": "search_documents",
    "search_doc": "search_documents",
    "search": "search_documents",
    "documentation_search": "search_documents",
    "doc_search": "search_documents",
    "lookup": "search_documents",
    "python": "run_python",
    "execute_python": "run_python",
    "python_exec": "run_python",
    "run": "run_python",
    "execute": "run_python",
    "validate": "validate_yaml",
    "yaml_validate": "validate_yaml",
    "yaml": "validate_yaml",
    "check_yaml": "validate_yaml",
    "validate_manifest": "validate_yaml",
    "check_manifest": "validate_yaml",
}

CANONICAL_TOOLS = {"search_documents", "run_python", "validate_yaml"}

logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "msg": "%(message)s"}',
)
logger = logging.getLogger(__name__)
logger.info(f"Agent initializing. INFERENCE_URL={INFERENCE_URL}")


def normalize_tool_name(name: str, canonical_tools: set[str] | None = None) -> str:
    """
    Map common LLM misspellings to canonical tool names.
    canonical_tools defaults to the static set if not provided.
    """
    if canonical_tools is None:
        canonical_tools = CANONICAL_TOOLS
    if name in canonical_tools:
        return name
    lowered = name.lower().strip()
    if lowered in canonical_tools:
        return lowered
    if lowered in TOOL_NAME_ALIASES:
        corrected = TOOL_NAME_ALIASES[lowered]
        logger.info(f"Tool name aliased: '{name}' -> '{corrected}'")
        return corrected
    return name


# ---------- Parser ----------

ACTION_PATTERN = re.compile(
    r"Action:\s*([\w_]+)\s*[\r\n]+\s*Action\s*Input:\s*(\{.*?\})",
    re.DOTALL | re.IGNORECASE,
)
FINAL_ANSWER_PATTERN = re.compile(
    r"Final\s*Answer:\s*(.*?)(?:\nThought:|\Z)",
    re.DOTALL | re.IGNORECASE,
)


def parse_output(text: str) -> dict:
    fa = FINAL_ANSWER_PATTERN.search(text)
    if fa:
        return {"type": "final", "answer": fa.group(1).strip()}

    act = ACTION_PATTERN.search(text)
    if act:
        tool = act.group(1).strip()
        input_str = act.group(2).strip()
        input_str = re.sub(r"^```(?:json)?\s*|\s*```$", "", input_str)
        try:
            args = json.loads(input_str)
            return {"type": "action", "tool": tool, "input": args}
        except json.JSONDecodeError as e:
            return {
                "type": "parse_error",
                "reason": f"Bad JSON in Action Input: {e}",
                "raw": input_str,
            }

    return {"type": "incomplete", "text": text}


# ---------- Logger ----------

class JSONLogger:
    def __init__(self):
        date_stamp = datetime.now().strftime("%Y%m%d")
        self.log_file = LOG_DIR / f"agent_{date_stamp}.jsonl"

    def log_session(
        self,
        session_id,
        question,
        answer,
        trace,
        latency_ms,
        input_tokens,
        output_tokens,
    ):
        n_tool_calls = sum(1 for t in trace if t["type"] == "tool_call")
        record = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "question": question,
            "answer": answer,
            "n_steps": len(trace),
            "n_tool_calls": n_tool_calls,
            "latency_ms": latency_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "trace": trace,
        }
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------- MCP client ----------

class MCPClient:
    def __init__(self, server_script: str):
        self.server_script = server_script
        self.session: ClientSession | None = None
        self._stdio_cm = None
        self._session_cm = None
        self.tools: list = []  # populated by discover_tools()

    async def start(self):
        params = StdioServerParameters(
            command=sys.executable,
            args=[self.server_script],
        )
        self._stdio_cm = stdio_client(params)
        read, write = await self._stdio_cm.__aenter__()
        self._session_cm = ClientSession(read, write)
        self.session = await self._session_cm.__aenter__()
        await self.session.initialize()
        logger.info("MCP client connected.")

    async def discover_tools(self) -> list:
        """
        Query the MCP server's tool list via list_tools().
        Per the MCP spec, this returns the server's exposed capabilities.
        """
        if not self.session:
            raise RuntimeError("MCP client not started")
        result = await self.session.list_tools()
        # mcp.types.ListToolsResult has .tools attribute
        self.tools = getattr(result, "tools", []) or []
        names = [t.name for t in self.tools]
        logger.info(f"MCP discovery: found {len(self.tools)} tools: {names}")
        return self.tools

    async def stop(self):
        try:
            if self._session_cm is not None:
                await self._session_cm.__aexit__(None, None, None)
            if self._stdio_cm is not None:
                await self._stdio_cm.__aexit__(None, None, None)
        except Exception as e:
            logger.warning(f"Error during MCP shutdown: {e}")

    async def call_tool(self, name: str, arguments: dict) -> str:
        if not self.session:
            raise RuntimeError("MCP client not started")
        result = await self.session.call_tool(name, arguments)
        if result.content:
            return "\n".join(
                c.text for c in result.content if hasattr(c, "text")
            )
        return ""


# ---------- Agent ----------

class ReActAgent:
    def __init__(
        self,
        inference_url: str = INFERENCE_URL,
        mcp_server_script: str = MCP_SERVER_SCRIPT,
        max_steps: int = MAX_STEPS,
    ):
        self.inference_url = inference_url
        self.max_steps = max_steps
        self.mcp = MCPClient(mcp_server_script)
        self.json_logger = JSONLogger()
        self.discovered_tools: list = []
        self.discovered_tool_names: set[str] = set()
        self.system_prompt: str = ""

    async def start(self):
        """Start MCP, discover tools, build the system prompt dynamically."""
        await self.mcp.start()
        self.discovered_tools = await self.mcp.discover_tools()
        self.discovered_tool_names = {t.name for t in self.discovered_tools}
        # Build the prompt with discovered tools (replaces hardcoded list)
        self.system_prompt = build_react_prompt(self.discovered_tools)
        logger.info(
            f"Agent ready. {len(self.discovered_tools)} tools available: "
            f"{sorted(self.discovered_tool_names)}"
        )

    async def stop(self):
        await self.mcp.stop()

    async def call_inference(
        self, messages: list, stop: list[str] | None = None
    ) -> dict:
        """
        Call the FastAPI inference server's /chat endpoint.
        Returns the full response dict including token counts.
        """
        async with httpx.AsyncClient(timeout=INFERENCE_TIMEOUT) as client:
            payload = {
                "messages": messages,
                "max_tokens": 512,
                "temperature": 0.2,
            }
            if stop:
                payload["stop"] = stop
            r = await client.post(f"{self.inference_url}/chat", json=payload)
            r.raise_for_status()
            return r.json()

    @staticmethod
    def _call_signature(tool: str, args: dict) -> str:
        try:
            args_str = json.dumps(args, sort_keys=True)
        except (TypeError, ValueError):
            args_str = str(args)
        return f"{tool}::{args_str}"

    async def run(
        self,
        question: str,
        history: list[dict] | None = None,
    ) -> tuple[str, list]:
        """
        Execute one ReAct turn.

        Args:
            question: the user's current query.
            history: optional list of past {"role", "content"} messages from
                     prior turns in this session. Will be inserted between the
                     system prompt and the current question.

        Returns:
            (final_answer, trace) tuple. The trace is the full ReAct trace
            for THIS turn only (history isn't included in the trace).
        """
        session_id = uuid.uuid4().hex[:8]
        start_time = time.time()

        # Build initial message list: system + history + current question
        messages: list[dict] = [
            {"role": "system", "content": self.system_prompt or build_react_prompt(self.discovered_tools)},
        ]
        if history:
            # Cap history to MAX_HISTORY_MESSAGES most recent
            capped = history[-MAX_HISTORY_MESSAGES:]
            messages.extend(capped)
        messages.append({"role": "user", "content": f"Question: {question}"})

        trace: list[dict] = []
        answer: str | None = None
        last_call_signature: str | None = None
        duplicate_call_count: int = 0
        last_observation: str | None = None

        # Aggregated token counts across all inference calls in this turn
        total_input_tokens = 0
        total_output_tokens = 0

        for step in range(self.max_steps):
            try:
                resp = await self.call_inference(
                    messages, stop=["Observation:"]
                )
            except Exception as e:
                trace.append({"step": step, "type": "inference_error", "content": str(e)})
                answer = f"Inference server error: {e}"
                break

            # Accumulate tokens
            total_input_tokens += resp.get("prompt_tokens", 0)
            total_output_tokens += resp.get("tokens", 0)

            model_out = resp.get("response", "")
            trace.append({"step": step, "type": "model_output", "content": model_out})
            parsed = parse_output(model_out)

            if parsed["type"] == "final":
                answer = parsed["answer"]
                trace.append({"step": step, "type": "final_answer", "content": answer})
                break

            elif parsed["type"] == "action":
                raw_tool_name = parsed["tool"]
                # Use discovered tools as the canonical set for aliasing
                canonical = self.discovered_tool_names or CANONICAL_TOOLS
                tool_name = normalize_tool_name(raw_tool_name, canonical)
                if tool_name != raw_tool_name:
                    trace.append({
                        "step": step,
                        "type": "tool_name_aliased",
                        "raw": raw_tool_name,
                        "corrected": tool_name,
                    })

                tool_args = parsed["input"]
                call_sig = self._call_signature(tool_name, tool_args)

                if call_sig == last_call_signature:
                    duplicate_call_count += 1
                    trace.append({
                        "step": step,
                        "type": "duplicate_call_detected",
                        "tool": tool_name,
                        "input": tool_args,
                        "consecutive_duplicates": duplicate_call_count,
                    })

                    if duplicate_call_count >= 2:
                        trace.append({"step": step, "type": "loop_break_forced_exit"})
                        if last_observation:
                            answer = (
                                "Based on the available information: "
                                + last_observation[:800]
                            )
                        else:
                            answer = (
                                "I was unable to converge on an answer "
                                "(detected a tool-call loop). Please try "
                                "rephrasing your question."
                            )
                        break
                    else:
                        messages.append({"role": "assistant", "content": model_out})
                        messages.append({
                            "role": "user",
                            "content": (
                                f"You just called {tool_name} with the same "
                                "arguments. Do NOT repeat. Either give Final "
                                "Answer or call a different tool."
                            ),
                        })
                        last_call_signature = call_sig
                        continue
                else:
                    duplicate_call_count = 0

                last_call_signature = call_sig

                trace.append({
                    "step": step,
                    "type": "tool_call",
                    "tool": tool_name,
                    "input": tool_args,
                })

                try:
                    observation = await self.mcp.call_tool(tool_name, tool_args)
                except Exception as e:
                    observation = f"Error calling tool '{tool_name}': {e}"

                if len(observation) > OBSERVATION_MAX_LEN:
                    observation = (
                        observation[:OBSERVATION_MAX_LEN] + "...[truncated]"
                    )

                last_observation = observation

                trace.append({
                    "step": step,
                    "type": "tool_result",
                    "content": observation,
                })

                messages.append({"role": "assistant", "content": model_out})
                messages.append({
                    "role": "user",
                    "content": f"Observation: {observation}",
                })

            elif parsed["type"] == "parse_error":
                trace.append({
                    "step": step,
                    "type": "parse_error",
                    "content": parsed["reason"],
                })
                messages.append({"role": "assistant", "content": model_out})
                messages.append({
                    "role": "user",
                    "content": (
                        f"Your Action Input was not valid JSON: {parsed['reason']}. "
                        "Retry with valid JSON on one line."
                    ),
                })

            else:  # incomplete
                trace.append({
                    "step": step,
                    "type": "incomplete_output",
                    "content": parsed["text"],
                })
                if len(parsed["text"].strip()) > 20:
                    answer = parsed["text"].strip()
                    break
                messages.append({"role": "assistant", "content": model_out})
                messages.append({
                    "role": "user",
                    "content": (
                        "I didn't see an Action or Final Answer. "
                        "Please respond using the format from the instructions."
                    ),
                })

        if answer is None:
            if last_observation:
                answer = "Based on what I found: " + last_observation[:600]
            else:
                answer = (
                    "I couldn't complete the reasoning within the allowed steps. "
                    "Try rephrasing your question."
                )
            trace.append({"step": self.max_steps, "type": "max_steps_exceeded"})

        latency_ms = int((time.time() - start_time) * 1000)
        self.json_logger.log_session(
            session_id,
            question,
            answer,
            trace,
            latency_ms,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
        )

        return answer, trace
