import logging
import os
import time
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
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
    version="0.2.1",
    description="FastAPI wrapper around Ollama serving fine-tuned Phi-3-mini.",
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
    tokens: int
    tokens_per_sec: float


def calc_throughput(eval_count: int, eval_duration_ns: int) -> float:
    if eval_duration_ns <= 0:
        return 0.0
    return round((eval_count / eval_duration_ns) * 1e9, 1)


def build_options(temperature: float, max_tokens: int, stop: Optional[list[str]]) -> dict:
    opts: dict = {"temperature": temperature, "num_predict": max_tokens}
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
    eval_duration_ns = data.get("eval_duration", 1)
    tps = calc_throughput(eval_count, eval_duration_ns)

    logger.info(
        f'generate model={req.model} latency_ms={latency_ms} '
        f'tokens={eval_count} tps={tps} stop={req.stop}'
    )

    return InferenceResponse(
        response=data.get("response", ""),
        model=req.model,
        latency_ms=latency_ms,
        tokens=eval_count,
        tokens_per_sec=tps,
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
    eval_duration_ns = data.get("eval_duration", 1)
    tps = calc_throughput(eval_count, eval_duration_ns)

    logger.info(
        f'chat model={req.model} latency_ms={latency_ms} '
        f'tokens={eval_count} tps={tps} stop={req.stop}'
    )

    return InferenceResponse(
        response=data.get("message", {}).get("content", ""),
        model=req.model,
        latency_ms=latency_ms,
        tokens=eval_count,
        tokens_per_sec=tps,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
