"""
FastAPI inference server wrapping Ollama.

Endpoints:
  GET  /health         - service health check
  GET  /models         - list available Ollama models
  POST /generate       - generate from prompt (single response, non-streaming)
  POST /generate/stream - generate from prompt with token-by-token streaming (SSE)
  POST /chat           - chat with messages array (multi-turn, non-streaming)

Streaming uses Server-Sent Events (SSE) format. Each event is JSON like:
  data: {"token": "Hello", "done": false}
  data: {"token": "", "done": true, "tokens": 42, "tokens_per_sec": 87.3}

Environment variables:
  OLLAMA_URL    - Ollama server base URL (default: http://localhost:11434)
  DEFAULT_MODEL - default model name (default: phi3-kubernetes)
"""

import json
import logging
import os
import time
from collections.abc import AsyncIterator
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "msg": "%(message)s"}',
)
logger = logging.getLogger(__name__)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "phi3-kubernetes")

logger.info(f"Inference server starting. OLLAMA_URL={OLLAMA_URL}")

app = FastAPI(
    title="Kubernetes Assistant Inference Server",
    version="0.4.0",
    description="FastAPI wrapper around Ollama serving fine-tuned Phi-3-mini. Streaming + prompt token tracking.",
)


class GenerateRequest(BaseModel):
    prompt: str
    model: Optional[str] = DEFAULT_MODEL
    temperature: Optional[float] = 0.4
    max_tokens: Optional[int] = 512
    stop: Optional[list[str]] = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    model: Optional[str] = DEFAULT_MODEL
    temperature: Optional[float] = 0.4
    max_tokens: Optional[int] = 512
    stop: Optional[list[str]] = None


class InferenceResponse(BaseModel):
    response: str
    model: str
    latency_ms: int
    tokens: int            # output tokens (eval_count from Ollama)
    prompt_tokens: int     # input tokens (prompt_eval_count from Ollama)
    tokens_per_sec: float


def calc_throughput(eval_count: int, eval_duration_ns: int) -> float:
    if eval_duration_ns <= 0:
        return 0.0
    return round((eval_count / eval_duration_ns) * 1e9, 1)


def build_options(temperature: float, max_tokens: int, stop: Optional[list[str]]) -> dict:
    opts: dict = {"temperature": temperature, "num_predict": max_tokens, "keep_alive": "10m"}
    if stop:
        opts["stop"] = stop
    return opts


@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{OLLAMA_URL}/api/tags", timeout=5.0)
            r.raise_for_status()
        return {"status": "ok", "ollama": "reachable", "ollama_url": OLLAMA_URL}
    except Exception as e:
        logger.error(f"health check failed: {e}")
        raise HTTPException(503, f"Ollama unreachable: {e}")


@app.get("/models")
async def list_models():
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{OLLAMA_URL}/api/tags", timeout=10.0)
        r.raise_for_status()
    data = r.json()
    return {"models": [m["name"] for m in data.get("models", [])]}


@app.post("/generate", response_model=InferenceResponse)
async def generate(req: GenerateRequest):
    """Non-streaming generation. Returns the full response when complete."""
    payload = {
        "model": req.model,
        "prompt": req.prompt,
        "stream": False,
        "options": build_options(req.temperature, req.max_tokens, req.stop),
    }
    start = time.time()
    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            r = await client.post(f"{OLLAMA_URL}/api/generate", json=payload)
            r.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise HTTPException(e.response.status_code, e.response.text)
    except Exception as e:
        raise HTTPException(503, f"Ollama error: {e}")

    latency_ms = int((time.time() - start) * 1000)
    data = r.json()
    eval_count = data.get("eval_count", 0)
    prompt_eval_count = data.get("prompt_eval_count", 0)
    eval_duration_ns = data.get("eval_duration", 1)
    tps = calc_throughput(eval_count, eval_duration_ns)

    logger.info(
        f'generate model={req.model} latency_ms={latency_ms} '
        f'tokens={eval_count} prompt_tokens={prompt_eval_count} tps={tps} stop={req.stop}'
    )

    return InferenceResponse(
        response=data.get("response", ""),
        model=req.model,
        latency_ms=latency_ms,
        tokens=eval_count,
        prompt_tokens=prompt_eval_count,
        tokens_per_sec=tps,
    )


async def stream_ollama(payload: dict) -> AsyncIterator[str]:
    """
    Forward Ollama's NDJSON stream as Server-Sent Events.
    """
    start = time.time()
    tokens_total = 0
    prompt_tokens = 0
    eval_duration_ns = 0

    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            async with client.stream("POST", f"{OLLAMA_URL}/api/generate", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if chunk.get("done", False):
                        tokens_total = chunk.get("eval_count", tokens_total)
                        prompt_tokens = chunk.get("prompt_eval_count", 0)
                        eval_duration_ns = chunk.get("eval_duration", 0)
                        tps = calc_throughput(tokens_total, eval_duration_ns)
                        final = {
                            "token": "",
                            "done": True,
                            "tokens": tokens_total,
                            "prompt_tokens": prompt_tokens,
                            "tokens_per_sec": tps,
                            "latency_ms": int((time.time() - start) * 1000),
                        }
                        yield f"data: {json.dumps(final)}\n\n"
                    else:
                        token = chunk.get("response", "")
                        if token:
                            payload_out = {"token": token, "done": False}
                            yield f"data: {json.dumps(payload_out)}\n\n"

    except httpx.HTTPStatusError as e:
        error = {"error": f"Ollama HTTP {e.response.status_code}", "done": True}
        yield f"data: {json.dumps(error)}\n\n"
    except Exception as e:
        error = {"error": str(e), "done": True}
        yield f"data: {json.dumps(error)}\n\n"


@app.post("/generate/stream")
async def generate_stream(req: GenerateRequest):
    """Token-by-token streaming generation via Server-Sent Events."""
    payload = {
        "model": req.model,
        "prompt": req.prompt,
        "stream": True,
        "options": build_options(req.temperature, req.max_tokens, req.stop),
    }
    logger.info(f"generate/stream model={req.model} prompt_len={len(req.prompt)}")

    return StreamingResponse(
        stream_ollama(payload),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/chat", response_model=InferenceResponse)
async def chat(req: ChatRequest):
    payload = {
        "model": req.model,
        "messages": [{"role": m.role, "content": m.content} for m in req.messages],
        "stream": False,
        "options": build_options(req.temperature, req.max_tokens, req.stop),
    }
    start = time.time()
    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            r = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)
            r.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise HTTPException(e.response.status_code, e.response.text)
    except Exception as e:
        raise HTTPException(503, f"Ollama error: {e}")

    latency_ms = int((time.time() - start) * 1000)
    data = r.json()
    eval_count = data.get("eval_count", 0)
    prompt_eval_count = data.get("prompt_eval_count", 0)
    eval_duration_ns = data.get("eval_duration", 1)
    tps = calc_throughput(eval_count, eval_duration_ns)

    logger.info(
        f'chat model={req.model} latency_ms={latency_ms} '
        f'tokens={eval_count} prompt_tokens={prompt_eval_count} tps={tps} stop={req.stop}'
    )

    return InferenceResponse(
        response=data.get("message", {}).get("content", ""),
        model=req.model,
        latency_ms=latency_ms,
        tokens=eval_count,
        prompt_tokens=prompt_eval_count,
        tokens_per_sec=tps,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
