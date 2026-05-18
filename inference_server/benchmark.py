import json
import statistics
import time
from pathlib import Path

import httpx

OLLAMA_URL = "http://localhost:11434"

# Models registered in Ollama (must match what you created)
MODELS_TO_TEST = ["phi3-kubernetes", "phi3-kubernetes-q8"]

# Test prompts covering different K8s topics
BENCHMARK_PROMPTS = [
    "What does kubectl rollout undo do?",
    "How do I create a ConfigMap from a file?",
    "Explain the difference between a Deployment and a StatefulSet.",
    "Write a YAML for a Service that exposes a Deployment on port 80.",
    "How do I view logs from a specific container in a pod with multiple containers?",
    "What is a HorizontalPodAutoscaler and how does it work?",
    "Show me a kubectl command to drain a node for maintenance.",
    "How can I troubleshoot a pod stuck in CrashLoopBackOff?",
]


def benchmark_model(model: str, prompts: list[str]) -> dict:
    """Run all prompts through a model. Return aggregated stats."""
    print(f"\n>>> Benchmarking {model}")
    per_prompt = []

    with httpx.Client(timeout=180.0) as client:
        # Warmup: first request after model load is always slower
        print("  warmup...")
        try:
            client.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": model,
                    "prompt": "Hello",
                    "stream": False,
                    "options": {"num_predict": 10},
                },
            )
        except Exception as e:
            print(f"  WARNING: warmup failed: {e}")

        for i, prompt in enumerate(prompts, 1):
            print(f"  [{i}/{len(prompts)}] {prompt[:60]}...")
            start = time.time()
            r = client.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": 256, "temperature": 0.4},
                },
            )
            wall_s = time.time() - start
            r.raise_for_status()
            data = r.json()

            tokens = data.get("eval_count", 0)
            eval_dur_ns = data.get("eval_duration", 1)
            prompt_eval_dur_ns = data.get("prompt_eval_duration", 1)

            tps = (tokens / eval_dur_ns) * 1e9 if eval_dur_ns > 0 else 0.0
            ttft_ms = prompt_eval_dur_ns / 1e6

            per_prompt.append({
                "prompt": prompt,
                "wall_time_s": round(wall_s, 2),
                "ttft_ms": round(ttft_ms, 1),
                "tokens_generated": tokens,
                "tokens_per_sec": round(tps, 1),
                "response_preview": data.get("response", "")[:200],
            })

    return {
        "model": model,
        "n_prompts": len(prompts),
        "total_wall_s": round(sum(p["wall_time_s"] for p in per_prompt), 2),
        "mean_wall_s": round(statistics.mean(p["wall_time_s"] for p in per_prompt), 2),
        "mean_ttft_ms": round(statistics.mean(p["ttft_ms"] for p in per_prompt), 1),
        "mean_tps": round(statistics.mean(p["tokens_per_sec"] for p in per_prompt), 1),
        "total_tokens": sum(p["tokens_generated"] for p in per_prompt),
        "per_prompt": per_prompt,
    }


def main():
    print("=" * 72)
    print("Quantization Benchmark: Q4_K_M vs Q8_0")
    print("=" * 72)
    print(f"Prompts: {len(BENCHMARK_PROMPTS)}")
    print(f"Models:  {MODELS_TO_TEST}")

    results = {}
    for model in MODELS_TO_TEST:
        try:
            results[model] = benchmark_model(model, BENCHMARK_PROMPTS)
        except Exception as e:
            print(f"\n!!! Benchmark of {model} failed: {e}")
            results[model] = {"error": str(e)}

    # Summary
    print("\n" + "=" * 72)
    print("Summary")
    print("=" * 72)
    print(
        f"{'Model':<28} {'Mean TTFT (ms)':<16} "
        f"{'Mean tok/s':<14} {'Total time (s)':<15}"
    )
    print("-" * 72)
    for model, stats in results.items():
        if "error" in stats:
            print(f"{model:<28} ERROR: {stats['error']}")
            continue
        print(
            f"{model:<28} {stats['mean_ttft_ms']:<16.1f} "
            f"{stats['mean_tps']:<14.1f} {stats['total_wall_s']:<15.1f}"
        )

    # Persist
    out = Path("benchmark_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {out}")


if __name__ == "__main__":
    main()
