# Engineering Notes

This document covers the written explanations called for by the assessment rubric:
dataset/training decisions (Task 1), agent failure modes (Task 4), and CI/CD
design choices (Task 6).

---

## Task 1 — Dataset choice, LoRA decisions, and evaluation analysis

**Domain choice.** Kubernetes Q&A was chosen for three reasons. First, it has
a deep public corpus (the official docs, kubernetes.io blog, CNCF talks) that
is rich in both terminology and concrete commands — useful for fine-tuning a
small model that needs grounding. Second, the answers are typically short and
verifiable (`kubectl scale deployment …`), which makes ROUGE-style automated
evaluation meaningful. Third, it fits naturally with the MCP tool set: a
domain that benefits from YAML validation and configuration computation maps
directly onto `validate_yaml` and `run_python`.

The dataset was curated from public Kubernetes documentation and adapted into
300 instruction-response pairs in ChatML format. Splits: 240 train / 30 val /
30 held-out test. No question text overlaps across splits.

**LoRA hyperparameters.** The fine-tune used QLoRA on Phi-3-mini-4k-instruct
with rank `r=16`, `alpha=32` (alpha/r = 2.0 is the conventional ratio),
target modules `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`,
learning rate `2e-4`, 3 epochs, batch size 2 with gradient accumulation 4
(effective batch 8). Rank 16 is the standard starting point for a 3.8B-param
model on a domain dataset of this size: high enough to capture domain phrasing,
low enough to avoid overfitting on 240 train samples. Learning rate `2e-4`
follows the Hugging Face PEFT/Unsloth default range. The full configuration
is reproducible via `training/train.py` and the saved adapter at
`training/output/adapter`.

**Evaluation results.** On the 30 held-out samples, the fine-tuned
`phi3-kubernetes` reaches ROUGE-L `0.1622` versus base `phi3:mini` at
`0.1382` — a `+17.4%` relative improvement. The fine-tune is also roughly
2× faster end-to-end because the model emits fewer extraneous tokens before
reaching the answer.

**Honest limitations.** ROUGE-L of 0.16 is modest in absolute terms. ROUGE
penalises lexical paraphrase even when the answer is correct (e.g., the model
saying "horizontal scaling" where the reference says "increasing replicas"),
so absolute scores understate quality on this domain. A larger held-out set
(100+ samples) and LLM-as-judge scoring would give a more stable signal. The
fine-tune does not change the model's underlying tool-use reliability — that
is addressed separately in Task 4.

---

## Task 4 — Agent failure modes and mitigations

During development I observed three distinct failure modes in the ReAct loop
running against Phi-3-mini. None are framework bugs — they are characteristic
of small models executing free-form ReAct prompts.

**Failure 1 — Skipping `Action:` emission.** Most common. The model produces
a `Thought:` then writes the final answer as prose, never emitting an
`Action:` line. The parser sees no tool call and returns the prose as the
final answer. Concrete example: when asked "validate this YAML", the model
sometimes responds with a generic explanation of what a Pod is, then writes
plausible-looking kubectl commands as if it had run them.
Mitigation: the system prompt in `agent/prompts.py` includes two complete
ReAct examples (validate_yaml and run_python) so the format is in-context.
A `MAX_STEPS=3` ceiling in `agent/agent.py` prevents runaway loops, and a
post-hoc check (`tool_calls_used`) records whether any tool was actually
invoked, so the orchestration layer can log this for analysis.

**Failure 2 — Tool name aliasing.** The model sometimes invents tool names
that resemble but don't match the declared tools (e.g. `yaml_validate`
instead of `validate_yaml`, `python_run` instead of `run_python`).
Mitigation: the agent's tool dispatcher validates the action name against
the registry of discovered MCP tools. Unknown names produce a structured
error observation that the model can recover from on the next step
(`agent/agent.py:_invoke_tool`).

**Failure 3 — Same-tool repetition loops.** On compound queries the model
sometimes calls `run_python` repeatedly with slight variations rather than
chaining a different tool. Mitigation: the max-step ceiling caps cost. The
multi-tool chain example in `scripts/demo_multi_tool_chain.py` uses an
explicit "TASK 1 / TASK 2" prompt to demonstrate that the agent CAN chain
tools when given enough structure — the failure is in the model's
self-direction, not the tool plumbing.

**General mitigation.** Each run is logged as structured JSON to
`logs/agent_*.jsonl` with full trace, tool calls, latencies, and token counts.
This makes the failure modes auditable rather than invisible. The
prioritised production fix is constrained JSON output via Ollama's
`format=json` (replaces free-form ReAct with grammar-guided generation),
estimated to lift multi-tool reliability from ~50% to ~85% on the same model.

---

## Task 6 — CI/CD design choices

**Self-hosted runner for evaluation.** The eval stage runs on a self-hosted
Windows runner with GPU access rather than GitHub-hosted Linux. This is the
key trade-off in the pipeline: GitHub-hosted runners have no GPU and would
either (a) run eval against a cloud Ollama (introduces external dependency
and cost), or (b) skip GPU-bound eval (loses the regression check). The
self-hosted runner solves this but introduces operational complexity — the
runner machine must stay available and the workflow has to handle the
runner-host network correctly (FastAPI on localhost, Ollama on localhost,
no Docker between them).

**Security model.** Eval and deploy stages are gated to `push` events on
`main` only. PRs from external collaborators run lint and Docker build
checks but not eval/deploy, preventing untrusted code from executing on
the self-hosted runner. The repository setting "Require approval for all
outside collaborators" is enforced. Secrets (`GHCR_TOKEN` for container
push) are stored as GitHub Actions secrets and never logged. The required
secrets are documented in the README.

**Threshold gate.** The eval stage asserts ROUGE-L > `0.14`. This value is
deliberately conservative: above the base `phi3:mini` score of `0.138` so
the gate would block a regression to base behavior, but well below the
current fine-tuned score of `0.162` so noise on the small 30-sample test
set doesn't cause false failures. If future fine-tunes push the score
higher, the threshold should be tightened.

**Multi-stage Docker.** Three images are built (inference, mcp,
orchestration), all using a multi-stage Dockerfile that copies wheels in
build stage and discards build tools in the runtime stage. ChromaDB
embeddings are baked into the MCP image at build time so cold-start does
not pay the embedding cost. Image SHAs are tagged with the commit SHA
plus `latest`.

**Rollback.** The deploy job records the previous image SHA before pulling
new images. If post-deploy `/health` fails, the previous image is re-tagged
to `latest` and the container restarted. Rollback was tested manually by
intentionally pushing an inference image with a broken entrypoint;
recovery completed in under 60 seconds.