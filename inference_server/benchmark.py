"""
Benchmark Q4_K_M vs Q8_0 quantizations of the Phi-3-mini fine-tuned model.

Measures per-model:
  - Time to first token (TTFT, ms)
  - Throughput (tokens/sec)
  - Total latency (ms)
  - Peak system RAM during runs (MB)
  - Peak GPU VRAM during runs (MB, if NVIDIA GPU + pynvml installed)

Run:
    python inference_server/benchmark.py

Writes inference_server/benchmark_results.json with all measurements.
"""

import json
import os
import statistics
import sys
import threading
import time
from pathlib import Path

import httpx

# Optional: GPU VRAM tracking via pynvml
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
    GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:
    GPU_AVAILABLE = False
    GPU_HANDLE = None

# Optional: process RAM tracking via psutil
try:
    import psutil
    PSUTIL_AVAILABLE = True
except Exception:
    PSUTIL_AVAILABLE = False

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
RESULTS_FILE = Path(__file__).parent / "benchmark_results.json"

# Models to benchmark
MODELS = {
    "phi3-kubernetes-q4": "phi3-kubernetes",      # Q4_K_M (deployed model)
    "phi3-kubernetes-q8": "phi3-kubernetes-q8",   # Q8_0 (high-quality variant)
}

PROMPTS = [
    "What is a Kubernetes Pod?",
    "How do I scale a Deployment to 5 replicas?",
    "Explain the difference between a Service and an Ingress.",
    "Write a YAML manifest for an Nginx Deployment with 3 replicas.",
    "What is the purpose of a PersistentVolumeClaim?",
    "How do RollingUpdates work in Deployments?",
    "Describe the role of kube-proxy.",
    "What is a StatefulSet used for?",
]


# ---------- Resource sampling ----------

class ResourceMonitor:
    """
    Background thread sampling system RAM and GPU VRAM every 200ms.
    Tracks peak values during a benchmark run.
    """

    def __init__(self, sample_interval: float = 0.2):
        self.sample_interval = sample_interval
        self._running = False
        self._thread: threading.Thread | None = None
        self.peak_ram_mb = 0.0
        self.peak_vram_mb = 0.0
        self._proc = psutil.Process() if PSUTIL_AVAILABLE else None

    def _sample(self):
        while self._running:
            # System-wide RAM used (in MB). Note: this captures total system
            # RAM not just our process, since Ollama runs in a separate process.
            if PSUTIL_AVAILABLE:
                try:
                    mem = psutil.virtual_memory()
                    used_mb = (mem.total - mem.available) / (1024 * 1024)
                    if used_mb > self.peak_ram_mb:
                        self.peak_ram_mb = used_mb
                except Exception:
                    pass

            # GPU VRAM (in MB) for device 0
            if GPU_AVAILABLE and GPU_HANDLE is not None:
                try:
                    info = pynvml.nvmlDeviceGetMemoryInfo(GPU_HANDLE)
                    used_mb = info.used / (1024 * 1024)
                    if used_mb > self.peak_vram_mb:
                        self.peak_vram_mb = used_mb
                except Exception:
                    pass

            time.sleep(self.sample_interval)

    def start(self):
        self.peak_ram_mb = 0.0
        self.peak_vram_mb = 0.0
        self._running = True
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        return {
            "peak_ram_mb": round(self.peak_ram_mb, 1),
            "peak_vram_mb": round(self.peak_vram_mb, 1),
        }


# ---------- Streaming benchmark ----------

def benchmark_one(model_name: str, prompt: str) -> dict:
    """
    Time a single prompt with streaming, capturing TTFT and throughput.
    """
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": True,
        "options": {"temperature": 0.4, "num_predict": 200},
    }

    start = time.time()
    first_token_time: float | None = None
    n_tokens = 0
    eval_duration_ns = 0

    with httpx.stream("POST", f"{OLLAMA_URL}/api/generate", json=payload, timeout=180.0) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue

            if first_token_time is None and chunk.get("response"):
                first_token_time = time.time()

            if chunk.get("done"):
                n_tokens = chunk.get("eval_count", n_tokens)
                eval_duration_ns = chunk.get("eval_duration", 0)
                break

    end = time.time()
    ttft_ms = int((first_token_time - start) * 1000) if first_token_time else 0
    total_ms = int((end - start) * 1000)
    tps = round((n_tokens / eval_duration_ns) * 1e9, 1) if eval_duration_ns > 0 else 0.0

    return {
        "prompt": prompt[:50] + ("..." if len(prompt) > 50 else ""),
        "ttft_ms": ttft_ms,
        "total_ms": total_ms,
        "tokens": n_tokens,
        "tokens_per_sec": tps,
    }


def benchmark_model(label: str, model_name: str) -> dict:
    print(f"\n{'=' * 60}")
    print(f"Benchmarking: {label} ({model_name})")
    print(f"{'=' * 60}")

    monitor = ResourceMonitor()
    monitor.start()

    # Warmup call (loads model into RAM/VRAM; not measured)
    print("  Warmup call...")
    try:
        benchmark_one(model_name, "Hello.")
    except Exception as e:
        monitor.stop()
        print(f"  Warmup failed: {e}")
        return {"label": label, "model": model_name, "error": str(e)}

    # Benchmark runs
    results = []
    for i, prompt in enumerate(PROMPTS, 1):
        print(f"  [{i}/{len(PROMPTS)}] {prompt[:50]}...")
        try:
            r = benchmark_one(model_name, prompt)
            results.append(r)
            print(f"      TTFT {r['ttft_ms']}ms · {r['tokens']} tokens · {r['tokens_per_sec']} tok/s")
        except Exception as e:
            print(f"      ERROR: {e}")
            results.append({"prompt": prompt[:50], "error": str(e)})

    peaks = monitor.stop()

    # Aggregate
    successful = [r for r in results if "error" not in r]
    ttfts = [r["ttft_ms"] for r in successful]
    tpss = [r["tokens_per_sec"] for r in successful]
    totals = [r["total_ms"] for r in successful]

    summary = {
        "label": label,
        "model": model_name,
        "n_prompts": len(PROMPTS),
        "n_successful": len(successful),
        "ttft_ms_mean": round(statistics.mean(ttfts), 1) if ttfts else 0,
        "ttft_ms_median": round(statistics.median(ttfts), 1) if ttfts else 0,
        "tokens_per_sec_mean": round(statistics.mean(tpss), 1) if tpss else 0,
        "total_ms_sum": sum(totals),
        "peak_ram_mb": peaks["peak_ram_mb"],
        "peak_vram_mb": peaks["peak_vram_mb"],
        "per_prompt": results,
    }

    print(f"\n  Summary for {label}:")
    print(f"    Mean TTFT:        {summary['ttft_ms_mean']} ms")
    print(f"    Mean throughput:  {summary['tokens_per_sec_mean']} tok/s")
    print(f"    Total time:       {summary['total_ms_sum'] / 1000:.1f}s for {len(PROMPTS)} prompts")
    print(f"    Peak RAM:         {summary['peak_ram_mb']} MB")
    if GPU_AVAILABLE:
        print(f"    Peak VRAM:        {summary['peak_vram_mb']} MB")
    else:
        print("    Peak VRAM:        n/a (no NVIDIA GPU detected)")

    return summary


def main():
    print("=" * 60)
    print("Phi-3 Kubernetes Quantization Benchmark")
    print("=" * 60)
    print(f"Ollama URL:    {OLLAMA_URL}")
    print(f"psutil:        {'available' if PSUTIL_AVAILABLE else 'NOT INSTALLED'}")
    print(f"GPU:           {'NVIDIA detected via pynvml' if GPU_AVAILABLE else 'not available (CPU-only mode)'}")
    print()

    if not PSUTIL_AVAILABLE:
        print("WARNING: psutil not installed. RAM measurements will be skipped.")
        print("         Install with: pip install psutil")
    if not GPU_AVAILABLE:
        print("INFO: pynvml not installed or no NVIDIA GPU. VRAM will be skipped.")
        print("      Install with: pip install nvidia-ml-py")
    print()

    # Verify Ollama reachable
    try:
        httpx.get(f"{OLLAMA_URL}/api/tags", timeout=5.0).raise_for_status()
    except Exception as e:
        print(f"ERROR: Ollama unreachable at {OLLAMA_URL}: {e}")
        print("Make sure Ollama is running.")
        sys.exit(1)

    all_results = {}
    for label, model in MODELS.items():
        try:
            all_results[label] = benchmark_model(label, model)
        except Exception as e:
            print(f"FAILED to benchmark {label}: {e}")
            all_results[label] = {"label": label, "model": model, "error": str(e)}

    # Comparison summary
    print("\n" + "=" * 60)
    print("Comparison")
    print("=" * 60)

    if "phi3-kubernetes-q4" in all_results and "phi3-kubernetes-q8" in all_results:
        q4 = all_results["phi3-kubernetes-q4"]
        q8 = all_results["phi3-kubernetes-q8"]

        if "error" not in q4 and "error" not in q8:
            print(f"  {'Metric':<22} {'Q4_K_M':<12} {'Q8_0':<12} {'Q4 advantage':<15}")
            print(f"  {'-' * 60}")

            ttft_speedup = q8["ttft_ms_mean"] / q4["ttft_ms_mean"] if q4["ttft_ms_mean"] > 0 else 0
            tps_speedup = q4["tokens_per_sec_mean"] / q8["tokens_per_sec_mean"] if q8["tokens_per_sec_mean"] > 0 else 0

            print(f"  {'TTFT mean (ms)':<22} {q4['ttft_ms_mean']:<12} {q8['ttft_ms_mean']:<12} {ttft_speedup:.2f}x faster")
            print(f"  {'Throughput (tok/s)':<22} {q4['tokens_per_sec_mean']:<12} {q8['tokens_per_sec_mean']:<12} {tps_speedup:.2f}x faster")
            print(f"  {'Total time (ms)':<22} {q4['total_ms_sum']:<12} {q8['total_ms_sum']:<12}")
            print(f"  {'Peak RAM (MB)':<22} {q4['peak_ram_mb']:<12} {q8['peak_ram_mb']:<12}")
            if GPU_AVAILABLE:
                print(f"  {'Peak VRAM (MB)':<22} {q4['peak_vram_mb']:<12} {q8['peak_vram_mb']:<12}")

    # Save
    output = {
        "ollama_url": OLLAMA_URL,
        "gpu_available": GPU_AVAILABLE,
        "psutil_available": PSUTIL_AVAILABLE,
        "results": all_results,
    }
    RESULTS_FILE.write_text(json.dumps(output, indent=2))
    print(f"\nResults written to {RESULTS_FILE}")

    # Cleanup pynvml
    if GPU_AVAILABLE:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
