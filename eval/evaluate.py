import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import httpx
from rouge_score import rouge_scorer

# ---------- Configuration ----------
INFERENCE_URL = "http://localhost:8000"
PROJECT_ROOT = Path(__file__).parent.parent
TEST_FILE = PROJECT_ROOT / "data" / "test.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "eval"
OUTPUT_FILE = OUTPUT_DIR / "results.json"

MODELS = {
    "base_phi3_mini": "phi3:mini",
    "fine_tuned_phi3_kubernetes": "phi3-kubernetes",
}

MAX_TOKENS = 400
TEMPERATURE = 0.0  # deterministic for fair comparison
TIMEOUT_S = 120


# ---------- Data loading ----------

def load_test_samples() -> list[dict]:
    """Parse data/test.jsonl into {question, reference} pairs."""
    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"{TEST_FILE} not found. Run prepare_dataset.py first."
        )

    samples = []
    with open(TEST_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            messages = data.get("messages", [])
            user_msg = next(
                (m for m in messages if m.get("role") == "user"), None
            )
            assistant_msg = next(
                (m for m in messages if m.get("role") == "assistant"), None
            )
            if user_msg and assistant_msg:
                samples.append({
                    "question": user_msg["content"],
                    "reference": assistant_msg["content"],
                })
    return samples


# ---------- Inference ----------

def query_model(model: str, question: str) -> tuple[str, int]:
    """Query a model via the FastAPI server. Returns (response, latency_ms)."""
    payload = {
        "messages": [{"role": "user", "content": question}],
        "model": model,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }
    start = time.time()
    with httpx.Client(timeout=TIMEOUT_S) as client:
        r = client.post(f"{INFERENCE_URL}/chat", json=payload)
        r.raise_for_status()
    latency_ms = int((time.time() - start) * 1000)
    return r.json()["response"].strip(), latency_ms


# ---------- Evaluation ----------

def evaluate_sample(scorer, sample: dict) -> dict:
    """Score one test sample against both models."""
    result = {
        "question": sample["question"],
        "reference": sample["reference"],
    }

    for label, model_name in MODELS.items():
        try:
            pred, latency_ms = query_model(model_name, sample["question"])
            score = scorer.score(sample["reference"], pred)["rougeL"].fmeasure
            result[label] = {
                "prediction": pred,
                "rouge_l": round(score, 4),
                "latency_ms": latency_ms,
            }
            print(f"  {label:35s} rouge_l={score:.3f}  latency={latency_ms}ms")
        except Exception as e:
            print(f"  {label:35s} ERROR: {e}")
            result[label] = {"error": str(e)}

    return result


def aggregate(results: list[dict]) -> dict:
    """Compute summary stats across all results."""
    summary: dict = {"n_samples": len(results)}

    for label in MODELS.keys():
        scores = [
            r[label]["rouge_l"]
            for r in results
            if "rouge_l" in r.get(label, {})
        ]
        latencies = [
            r[label]["latency_ms"]
            for r in results
            if "latency_ms" in r.get(label, {})
        ]
        if scores:
            summary[label] = {
                "n_success": len(scores),
                "rouge_l_mean": round(statistics.mean(scores), 4),
                "rouge_l_median": round(statistics.median(scores), 4),
                "rouge_l_stdev": round(statistics.stdev(scores), 4) if len(scores) > 1 else 0,
                "latency_ms_mean": int(statistics.mean(latencies)) if latencies else 0,
            }
        else:
            summary[label] = {"n_success": 0, "rouge_l_mean": 0}

    # Improvement delta
    base = summary["base_phi3_mini"].get("rouge_l_mean", 0)
    ft = summary["fine_tuned_phi3_kubernetes"].get("rouge_l_mean", 0)
    summary["improvement"] = {
        "rouge_l_absolute_delta": round(ft - base, 4),
        "rouge_l_relative_pct": (
            round((ft - base) / base * 100, 2) if base > 0 else 0
        ),
    }
    return summary


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Minimum ROUGE-L for fine-tuned model. "
            "Exit code 1 if not met. Used in CI."
        ),
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Evaluation: base phi3:mini vs fine-tuned phi3-kubernetes")
    print("=" * 70)
    print(f"Test file:  {TEST_FILE}")
    print("Metric:     ROUGE-L (F-measure)")
    print(f"Models:     {list(MODELS.values())}")

    # Health check
    print("\nChecking inference server...")
    try:
        with httpx.Client(timeout=10) as client:
            r = client.get(f"{INFERENCE_URL}/health")
            r.raise_for_status()
        print(f"  OK: {r.json()}")
    except Exception as e:
        print(f"  FAIL: {e}")
        print(
            f"\nMake sure the FastAPI server is running at {INFERENCE_URL}\n"
            "  cd inference_server && python main.py"
        )
        sys.exit(2)

    # Load samples
    print("\nLoading samples...")
    samples = load_test_samples()
    print(f"  {len(samples)} samples loaded.\n")

    # Score each sample against each model
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    results = []
    for i, s in enumerate(samples, 1):
        print(f"[{i}/{len(samples)}] {s['question'][:80].replace(chr(10), ' ')}...")
        result = evaluate_sample(scorer, s)
        results.append(result)

    # Aggregate
    summary = aggregate(results)

    # Persist
    output = {"summary": summary, "per_sample": results}
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Samples evaluated:           {summary['n_samples']}")
    print()
    base = summary["base_phi3_mini"]
    ft = summary["fine_tuned_phi3_kubernetes"]
    print("Base phi3:mini")
    print(f"  ROUGE-L mean:              {base.get('rouge_l_mean', 0):.4f}")
    print(f"  ROUGE-L median:            {base.get('rouge_l_median', 0):.4f}")
    print(f"  Latency mean:              {base.get('latency_ms_mean', 0)}ms")
    print()
    print("Fine-tuned phi3-kubernetes")
    print(f"  ROUGE-L mean:              {ft.get('rouge_l_mean', 0):.4f}")
    print(f"  ROUGE-L median:            {ft.get('rouge_l_median', 0):.4f}")
    print(f"  Latency mean:              {ft.get('latency_ms_mean', 0)}ms")
    print()
    imp = summary["improvement"]
    sign = "+" if imp["rouge_l_absolute_delta"] >= 0 else ""
    print("Improvement (fine-tuned vs base):")
    print(f"  Absolute delta:            {sign}{imp['rouge_l_absolute_delta']:.4f}")
    print(f"  Relative:                  {sign}{imp['rouge_l_relative_pct']:+.2f}%")
    print()
    print(f"Detailed results saved to:   {OUTPUT_FILE}")

    # CI threshold check
    if args.threshold is not None:
        ft_score = ft.get("rouge_l_mean", 0)
        if ft_score < args.threshold:
            print(
                f"\nFAIL: Fine-tuned ROUGE-L {ft_score:.4f} < threshold "
                f"{args.threshold:.4f}"
            )
            sys.exit(1)
        else:
            print(
                f"\nPASS: Fine-tuned ROUGE-L {ft_score:.4f} >= threshold "
                f"{args.threshold:.4f}"
            )


if __name__ == "__main__":
    main()
