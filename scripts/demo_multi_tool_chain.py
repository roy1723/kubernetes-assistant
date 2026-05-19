"""
scripts/demo_multi_tool_chain.py

Demonstrates the agent's multi-tool chaining capability by bypassing the
Gradio router and invoking ReActAgent.run() directly with an explicit prompt.

Run from the project root:
    python scripts/demo_multi_tool_chain.py
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "agent"))

from agent import ReActAgent

PROMPT = (
    "I need you to do two tasks in this exact order:\n"
    "TASK 1: Use the search_documents tool with query 'HorizontalPodAutoscaler' "
    "to find documentation context about HPA.\n"
    "TASK 2: Use the run_python tool to execute "
    "`print(round(8 * 0.6, 2))` and capture the output.\n"
    "After both tools have returned observations, give a Final Answer that "
    "combines what you learned from the documentation with the computed value."
)

OUTPUT_PATH = PROJECT_ROOT / "docs" / "multi_tool_chain_trace.json"


async def main():
    print("=" * 60)
    print("Multi-tool chain demonstration")
    print("=" * 60)
    print(f"Prompt:\n{PROMPT}\n")
    print("Starting agent (initializing MCP, discovering tools)...")

    agent = ReActAgent()
    await agent.start()

    print(f"Agent ready. Tools available: {sorted(agent.discovered_tool_names)}")
    print("\nRunning chain...\n")

    try:
        answer, trace = await agent.run(PROMPT)
    finally:
        await agent.stop()

    print("=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)
    print(answer)
    print()

    tool_calls = [t for t in trace if t.get("type") == "tool_call"]
    tools_used = [t["tool"] for t in tool_calls]
    print("=" * 60)
    print("TRACE SUMMARY")
    print("=" * 60)
    print(f"Tool calls: {len(tool_calls)}")
    print(f"Tools invoked: {tools_used}")
    print(f"Total steps: {len(trace)}")

    chain_succeeded = (
        "search_documents" in tools_used and "run_python" in tools_used
    )
    if chain_succeeded:
        print("\n+ SUCCESS: agent chained search_documents -> run_python")
    else:
        print("\n- Chain not completed this run. Rerun, or see NOTES.md "
              "Task 4 for failure-mode analysis.")

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    record = {
        "timestamp": datetime.now().isoformat(),
        "prompt": PROMPT,
        "answer": answer,
        "tools_invoked": tools_used,
        "chain_succeeded": chain_succeeded,
        "trace": trace,
    }
    OUTPUT_PATH.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    print(f"\nTrace written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    os.environ.setdefault("INFERENCE_URL", "http://127.0.0.1:8000")
    asyncio.run(main())
