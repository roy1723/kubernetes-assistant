import asyncio
import os
import sys

# Ensure agent.py is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import ReActAgent

TEST_QUESTIONS = [
    # Test 1: should trigger search_documents
    "How do I roll back a deployment to the previous revision in Kubernetes?",

    # Test 2: should trigger validate_yaml
    "Is this YAML valid as a Kubernetes resource?\n"
    "apiVersion: v1\nkind: Service\nmetadata:\n  name: my-svc\n"
    "spec:\n  selector:\n    app: nginx\n  ports:\n    - port: 80",

    # Test 3: should trigger run_python
    "Calculate how many pods can fit on 5 nodes if each node has 16 GB RAM "
    "and each pod needs 2 GB. Show the math.",

    # Test 4: can be answered without any tool
    "What is the difference between a Deployment and a StatefulSet in one sentence?",
]


def print_trace(trace):
    print(f"\n--- Trace ({len(trace)} entries) ---")
    for entry in trace:
        t = entry.get("type", "?")
        if t == "model_output":
            preview = entry["content"][:300].replace("\n", " | ")
            print(f"  [model_output] {preview}")
        elif t == "tool_call":
            print(f"  [tool_call] {entry['tool']}({entry['input']})")
        elif t == "tool_result":
            preview = entry["content"][:300].replace("\n", " | ")
            print(f"  [tool_result] {preview}")
        elif t == "final_answer":
            print(f"  [final_answer] {entry['content'][:300]}")
        elif t == "parse_error":
            print(f"  [parse_error] {entry['content']}")
        elif t == "incomplete_output":
            preview = entry["content"][:200].replace("\n", " | ")
            print(f"  [incomplete] {preview}")
        elif t == "inference_error":
            print(f"  [inference_error] {entry['content']}")
        elif t == "max_steps_exceeded":
            print("  [max_steps_exceeded]")
        else:
            print(f"  [{t}] {str(entry)[:200]}")


async def main():
    agent = ReActAgent()
    print("Starting agent (launching MCP server subprocess)...")
    await agent.start()
    print("Agent ready.\n")

    try:
        for i, question in enumerate(TEST_QUESTIONS, 1):
            print("=" * 78)
            preview = question.split("\n")[0]
            print(f"TEST {i}: {preview[:100]}{'...' if len(preview) > 100 else ''}")
            print("=" * 78)

            try:
                answer, trace = await agent.run(question)
            except Exception as e:
                print(f"\n!!! Agent.run failed: {type(e).__name__}: {e}")
                continue

            print_trace(trace)
            print(f"\n--- Final Answer ---\n{answer}\n")

    finally:
        await agent.stop()
        print("Agent stopped.")


if __name__ == "__main__":
    asyncio.run(main())
