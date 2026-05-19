# Kubernetes Assistant

![CI/CD](https://github.com/roy1723/kubernetes-assistant/actions/workflows/ci.yml/badge.svg)

A local, end-to-end AI research assistant for Kubernetes Q&A. The pipeline fine-tunes Phi-3-mini on a self-curated Kubernetes dataset, serves it locally via Ollama behind a FastAPI inference layer, exposes three MCP tools (semantic search, sandboxed Python, YAML validation), and orchestrates everything through a ReAct agent fronted by a Gradio chat UI. The whole system ships with a four-stage CI/CD pipeline that lints, evaluates, builds Docker images to GHCR, and auto-deploys to a self-hosted runner.

**Detailed write-ups** (dataset choice, training decisions, failure modes, CI/CD design): see [`NOTES.md`](./NOTES.md).

---

## Headline results

### Fine-tuning improvement (Task 1)

| Metric         | Base `phi3:mini` | Fine-tuned `phi3-kubernetes` | Δ              |
| -------------- | ---------------- | ---------------------------- | -------------- |
| ROUGE-L        | 0.1382           | **0.1622**                   | **+17.4%**     |
| Avg latency    | 18,104 ms        | **9,272 ms**                 | **2× faster**  |

Evaluated on 30 held-out K8s Q&A samples not in train/val.

### Quantization benchmark (Task 2 — RTX 3050 Laptop, 4 GB VRAM)

| Metric                  | Q4_K_M (deployed)  | Q8_0           | Q4 advantage           |
| ----------------------- | ------------------ | -------------- | ---------------------- |
| TTFT (cold start)       | 2,692 ms           | 2,798 ms       | ~1× (no diff)          |
| Throughput              | **63.6 tok/s**     | 17.3 tok/s     | **3.7× faster**        |
| Total time (8 prompts)  | 36.6 s             | 84.3 s         | **2.3× faster**        |
| Peak RAM                | 11.6 GB            | 12.8 GB        | 1.2 GB lower           |
| Peak VRAM               | **3.4 GB**         | 3.6 GB         | 140 MB lower           |

Q4_K_M is chosen for deployment: ~4× throughput at the same VRAM ceiling, fits the 4 GB VRAM budget with headroom. Full results in `inference_server/benchmark_results.json`.

---

## Architecture

```
                            User browser
                                  │
                                  ▼
                  ┌─────────────────────────────┐
                  │   Gradio UI (port 7860)     │
                  │   - per-session gr.State    │
                  │   - history capped at 6     │
                  └──────────────┬──────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────────────┐
                  │  Router (hybrid keyword + LLM)       │
                  │  classify → casual / direct / tools  │
                  └────┬─────────┬─────────┬─────────────┘
                       │         │         │
              ┌────────┘         │         └──────────┐
              ▼                  ▼                    ▼
       ┌──────────────┐  ┌─────────────────┐  ┌──────────────────┐
       │   Casual     │  │  Direct path    │  │   ReAct Agent    │
       │  (static)    │  │  inference only │  │  Thought→Action  │
       └──────────────┘  └────────┬────────┘  │  →Observation    │
                                  │           │  loop (≤3 steps) │
                                  │           └────────┬─────────┘
                                  │                    │
                                  │                    │ stdio
                                  │                    ▼
                                  │           ┌────────────────────┐
                                  │           │   MCP Server       │
                                  │           │   - search_docs    │
                                  │           │   - run_python     │
                                  │           │   - validate_yaml  │
                                  │           └────────┬───────────┘
                                  │                    │
                                  │                    │
                                  ▼                    ▼
              ┌──────────────────────────────────────────────────┐
              │   FastAPI Inference Server (port 8000)           │
              │   /generate · /generate/stream · /chat · /health │
              └──────────────────────────┬───────────────────────┘
                                         │
                                         ▼
                          ┌──────────────────────────────┐
                          │   Ollama (port 11434)        │
                          │   phi3-kubernetes (Q4_K_M)   │
                          │   phi3-kubernetes-q8 (Q8_0)  │
                          │   phi3:mini (router only)    │
                          └──────────────────────────────┘
```

**Container boundaries** (Docker deploy): `ollama`, `mcp_server`, `inference_server`, `orchestration` run as four separate services on the same Docker network. The MCP server is invoked as a subprocess by the agent via stdio; it embeds ChromaDB with 211 K8s doc chunks built at image-build time.

---

## Quickstart

### Prerequisites

- **OS**: Linux, macOS, or Windows 10+
- **Docker**: Docker Desktop or Docker Engine with Compose v2
- **Hardware**: 16 GB RAM minimum, 4 GB free disk for Ollama models
- **GPU (optional)**: NVIDIA GPU with 4+ GB VRAM for fast inference (CPU mode also works, ~5× slower)

### One-command launch from pre-built images

```bash
# Public images on GHCR — no auth needed for pulls
docker compose -f docker-compose.deploy.yml up -d
```

This pulls 3 images from `ghcr.io/roy1723/kubernetes-assistant-*:latest` and starts the stack. Wait ~60 seconds for Ollama to download the model on first run.

Open <http://localhost:7860> for the Gradio chat UI.

### Build from source

```bash
git clone https://github.com/roy1723/kubernetes-assistant
cd kubernetes-assistant
docker compose up --build
```

First build is slow (~10-15 min) because the MCP image embeds ChromaDB + sentence-transformers. Subsequent builds use Docker layer cache.

### Native Python (for development)

```bash
python -m venv venv
source venv/bin/activate         # Linux/macOS
venv\Scripts\activate.bat        # Windows
pip install -r requirements.txt

# Terminal 1: Ollama (must be installed separately — ollama.com)
ollama serve

# Terminal 2: FastAPI inference server
python inference_server/main.py

# Terminal 3: Gradio UI (auto-starts agent + MCP via stdio subprocess)
python orchestration/app.py
```

Visit <http://localhost:7860>.

---

## Test the system

### Three example queries to try

1. **Casual** (keyword fast-path, no LLM):

   ```
   hi
   ```

2. **Direct** (fine-tuned model, no tools):

   ```
   What is the difference between a Deployment and a StatefulSet?
   ```

3. **Tools — YAML validation** (ReAct agent + `validate_yaml`):

   ```
   validate this YAML:
   apiVersion: v1
   kind: Pod
   metadata:
     name: test
   ```

### Verify observability

Per-request JSON logs accumulate in `logs/agent_YYYYMMDD.jsonl`. Each record contains:

```json
{
  "timestamp": "...",
  "session_id": "...",
  "question": "...",
  "answer": "...",
  "n_steps": 5,
  "n_tool_calls": 1,
  "latency_ms": 5376,
  "input_tokens": 2160,
  "output_tokens": 121,
  "trace": [...]
}
```

The `trace` field captures every Thought → Action → Observation → Final Answer step for full agent debuggability.

---

## CI/CD pipeline (Task 6)

The workflow at `.github/workflows/ci.yml` runs four jobs on every push to `main` and on every PR:

| Job              | Runner               | Steps                                                                |
| ---------------- | -------------------- | -------------------------------------------------------------------- |
| `lint-and-type`  | GitHub Ubuntu        | ruff + mypy across all source dirs                                   |
| `evaluate`       | Self-hosted (laptop) | Start FastAPI background, run eval, assert ROUGE-L ≥ 0.14, artifact  |
| `docker-build`   | GitHub Ubuntu        | Build 3 multi-stage images, push to GHCR with SHA + `latest` tags    |
| `deploy`         | Self-hosted (laptop) | Pull from GHCR, restart stack, health-check, rollback on failure     |

The eval and deploy jobs are gated to `push` events on the `main` branch — PRs only run lint and build, keeping fork-PR safety. The runner machine is configured to require interactive auth for external contributors via repository settings.

### Required secrets

| Secret name    | Used for                                              | Scope                                             |
| -------------- | ----------------------------------------------------- | ------------------------------------------------- |
| `GHCR_TOKEN`   | Push Docker images to GitHub Container Registry      | Classic PAT with `write:packages` permission only |

`GHCR_TOKEN` is also used during the deploy stage to authenticate `docker pull` on the runner. No other secrets are required. There are no API keys to manage — the model runs locally.

### Self-hosted runner setup

The eval and deploy jobs run on a Windows self-hosted runner. Setup instructions in `docs/runner-setup.md`. Key points:

- Runner runs interactively as the desktop user (NOT as a service) so it can talk to Docker Desktop and the user-mode Ollama daemon
- Both `phi3-kubernetes` and `phi3-kubernetes-q8` must be registered in the local Ollama instance
- The `actions-runner/_work` directory must be owned by the user account, with `git config --global --add safe.directory '*'` set for the runner user

---

## Known limitations

1. **ReAct reliability on small models.** Phi-3-mini sometimes hallucinates a Final Answer instead of emitting an `Action:` block — most visible on math/computation queries. Mitigations in place (loop detection, tool name aliasing, parse-error recovery) keep the failure mode bounded but don't eliminate it. **Detailed analysis in [NOTES.md → Task 4](./NOTES.md#task-4-agent-failure-modes-and-mitigations).**

2. **Multi-tool chains rarely succeed.** Chaining `search_documents` followed by `run_python` requires the model to make two distinct tool decisions in sequence. With Phi-3-mini, this works ~20% of the time. Each tool in isolation works ~80% (validate_yaml) or ~50% (run_python).

3. **Router misclassifies action-verb prompts.** Queries like "search the docs for X then compute Y" sometimes route to `direct` instead of `tools`, bypassing the agent entirely. The keyword fast-path covers YAML / "validate" / math+units triggers but doesn't pattern-match action verbs.

4. **Cold-start TTFT is high (~2.7 s).** First-token latency on the RTX 3050 includes model KV-cache initialization. Subsequent calls within ~10 min are faster (Ollama keeps the model loaded). The `keep_alive` option in inference requests holds the model warm.

5. **Docker deployment on Windows is CPU-only.** `nvidia-container-toolkit` isn't supported on Docker Desktop for Windows, so the deployed stack runs Ollama in CPU mode (~13 tok/s vs ~64 tok/s on native GPU). Native Python deployment uses the GPU. The Docker stack proves the deploy works; native is the demo path.

6. **GGUF files are large.** The Q4_K_M (2.3 GB) and Q8_0 (4 GB) GGUF files exceed GitHub's per-file limit and are not committed. The Q4_K_M is hosted on Hugging Face Hub; the Q8_0 is not hosted (regenerable locally from the LoRA adapter). See [Model weights](#model-weights) below.

---

## Prioritized improvement

**Replace Phi-3-mini with Llama-3-8B-Instruct (fine-tuned via the same QLoRA pipeline) as the deployed model.**

The single biggest limitation today is small-model unreliability with ReAct. A 8B parameter model would handle multi-step reasoning and tool chaining substantially better — multi-tool chain success would likely climb from ~20% to ~70%+, and the math-skip-tool failure mode would mostly disappear. The cost is VRAM: Q4_K_M of Llama-3-8B needs ~5 GB, which exceeds the RTX 3050's 4 GB. The fix is either GPU offloading (slower but works) or upgrading the deployment target to a 12 GB+ GPU. The training pipeline (`scripts/fine_tune_phi3.py`) is model-agnostic — Llama-3-8B would slot in by changing the base model identifier and adjusting LoRA rank/alpha empirically.

This is the highest-value upgrade because reliability gains propagate through every downstream task: routing accuracy, tool-call success, multi-tool chains, and final-answer quality.

---

## Model weights

**Hugging Face Hub**: <https://huggingface.co/shlbnrj/phi3-kubernetes>

| File                              | Hosted on HF? | Purpose                                         |
| --------------------------------- | ------------- | ----------------------------------------------- |
| `phi3_kubernetes_lora.zip`        | ✅ yes        | LoRA adapter weights (rank=16, alpha=32)        |
| `phi3-kubernetes-q4_k_m.gguf`     | ✅ yes        | Q4_K_M GGUF — deployed quantization, 2.16 GB    |
| `phi3-kubernetes-q8_0.gguf`       | ❌ not hosted | Q8_0 GGUF — benchmark only, ~4 GB               |

The Q8_0 GGUF is not hosted to keep the HF repo small. Its benchmark numbers are in `inference_server/benchmark_results.json`. It can be regenerated locally from the LoRA adapter using `llama-quantize` from `llama.cpp`.

Quick install for reviewers using Ollama:

```bash
ollama run hf.co/shlbnrj/phi3-kubernetes:Q4_K_M
```

The repo's `outputs/lora_adapters/` directory contains the adapter `.zip` locally but is `.gitignored` (>100 MB).

---

## Project layout

```
.
├── .github/workflows/ci.yml      # CI/CD pipeline (4 jobs)
├── data/
│   ├── train.jsonl               # 240 K8s Q&A samples
│   ├── val.jsonl                 # 30 samples
│   ├── test.jsonl                # 30 held-out samples
│   └── k8s_docs.json             # 211 doc chunks for ChromaDB
├── inference_server/
│   ├── main.py                   # FastAPI (port 8000)
│   ├── benchmark.py              # Q4 vs Q8 measurement
│   ├── benchmark_results.json    # Latest benchmark output
│   └── Modelfile, Modelfile.q8   # Ollama configs
├── mcp_server/
│   ├── server.py                 # MCP via stdio
│   ├── tools.py                  # 3 tool implementations
│   └── test_tools.py             # valid + invalid input tests
├── agent/
│   ├── agent.py                  # ReAct agent + MCPClient
│   └── prompts.py                # build_react_prompt(tools) — dynamic
├── orchestration/
│   ├── app.py                    # Gradio UI + session memory
│   └── router.py                 # Hybrid keyword+LLM classifier
├── eval/
│   ├── evaluate.py               # ROUGE-L on test set
│   └── results.json              # Eval output (CI artifact)
├── tests/test_smoke.py           # 10 sanity tests (run by CI)
├── docker/
│   ├── Dockerfile.{inference,mcp,orchestration}
│   └── requirements.*.txt
├── docs/
│   ├── runner-setup.md           # Self-hosted runner instructions
│   └── architecture.txt          # Detailed ASCII data-flow diagram
├── docker-compose.yml            # Local build
├── docker-compose.deploy.yml     # Pull from GHCR
├── ruff.toml                     # Lint config
├── mypy.ini                      # Type-check config
└── NOTES.md                      # Written explanations (Tasks 1, 4, 6)
```