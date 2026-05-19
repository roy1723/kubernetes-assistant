---
license: mit
language: en
tags:
  - kubernetes
  - phi-3
  - lora
  - qlora
  - gguf
base_model: microsoft/Phi-3-mini-4k-instruct
---

# phi3-kubernetes

Phi-3-mini fine-tuned on 300 Kubernetes Q&A samples from Stack Overflow using QLoRA. Built as part of an end-to-end Local Research Assistant pipeline. See the [project repo](https://github.com/roy1723/kubernetes-assistant) for full context, including dataset preparation, evaluation results, inference server, and CI/CD setup.

## Files

| File                              | Size       | Purpose                                          |
| --------------------------------- | ---------- | ------------------------------------------------ |
| `phi3_kubernetes_lora.zip`        | 106 MB     | LoRA adapter weights (rank=16, alpha=32)         |
| `phi3-kubernetes-q4_k_m.gguf`     | 2.16 GB    | Merged GGUF, 4-bit quant — **recommended deployment** |
| `phi3-kubernetes-q8_0.gguf`       | not hosted | Benchmarked locally; regenerable with `llama-quantize` from the LoRA adapter |

The Q8_0 quantization was benchmarked during development (see eval section below) but is not hosted here to keep the repo small. The Q4_K_M is the deployment-recommended variant: ~3.7× faster throughput at the same VRAM ceiling.

## Training

| Hyperparameter   | Value                  |
| ---------------- | ---------------------- |
| Base model       | microsoft/Phi-3-mini-4k-instruct |
| Fine-tune method | QLoRA (Unsloth)        |
| LoRA rank        | 16                     |
| LoRA alpha       | 32                     |
| Learning rate    | 2e-4 (cosine schedule) |
| Batch size       | 8 (effective 16 via grad accumulation) |
| Epochs           | 3                      |
| Hardware         | Colab T4 (16 GB VRAM)  |
| Dataset          | mcipriano/stackoverflow-kubernetes-questions (300 samples filtered) |

## Evaluation

Tested on 30 held-out K8s Q&A samples against the base `phi3:mini` model:

| Metric           | Base phi3:mini  | Fine-tuned phi3-kubernetes | Δ          |
| ---------------- | --------------- | -------------------------- | ---------- |
| ROUGE-L          | 0.1382          | **0.1622**                 | **+17.4%** |
| Avg latency      | 18,104 ms       | 9,272 ms                   | 2× faster  |

## Quantization benchmark (RTX 3050 Laptop, 4 GB VRAM)

| Quant     | TTFT (ms) | Throughput (tok/s) | Peak VRAM |
| --------- | --------- | ------------------ | --------- |
| Q4_K_M    | 2,692     | **63.6**           | 3.4 GB    |
| Q8_0      | 2,798     | 17.3               | 3.6 GB    |

Q4_K_M is ~3.7× faster at the same VRAM ceiling — chosen as the deployment default.

## Usage with Ollama

```bash
ollama run hf.co/shlbnrj/phi3-kubernetes:Q4_K_M
```

Or manually:

```bash
wget https://huggingface.co/shlbnrj/phi3-kubernetes/resolve/main/phi3-kubernetes-q4_k_m.gguf

cat > Modelfile <<'EOF'
FROM ./phi3-kubernetes-q4_k_m.gguf
PARAMETER temperature 0.4
PARAMETER num_ctx 4096
EOF

ollama create phi3-kubernetes -f Modelfile
ollama run phi3-kubernetes "What is a Pod?"
```

## License

MIT. The base model (Phi-3) is under the [Microsoft Research License](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/blob/main/LICENSE).

## Limitations

This is a small (3.8B parameter) model. It works well for direct K8s knowledge questions but is unreliable for multi-step tool-use scenarios (e.g., chaining a search with a Python computation). See the project's `NOTES.md` for detailed failure-mode analysis.