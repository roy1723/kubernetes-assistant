"""
orchestration/router.py

Hybrid query router for the Kubernetes Assistant.

Two-stage routing:
  1. Keyword fast-path (regex rules, O(microseconds))
  2. LLM classifier   (base phi3:mini, ~500ms) for ambiguous cases

Labels: casual / direct / tools

The router is stateless. Session history is managed in app.py and passed
into the agent/direct path after routing.

Environment variables:
  INFERENCE_URL - FastAPI inference server URL (default: http://localhost:8000)
"""

import logging
import os
import re

import httpx

logger = logging.getLogger(__name__)

DEFAULT_INFERENCE_URL = os.getenv("INFERENCE_URL", "http://localhost:8000")
DEFAULT_MODEL = "phi3:mini"

# ---------- Keyword patterns ----------

GREETING_PATTERN = re.compile(
    r"^\s*(hi|hello|hey|yo|greetings|sup|thanks|thank you|bye|goodbye)\b",
    re.IGNORECASE,
)
ASSISTANT_QUESTION_PATTERN = re.compile(
    r"\b(what can you do|how can you help|who are you|what are you|"
    r"introduce yourself|what is this|are you (an? |a )?bot)\b",
    re.IGNORECASE,
)
YAML_FIELDS_PATTERN = re.compile(
    r"\b(apiVersion|kind|metadata|spec)\s*:",
    re.IGNORECASE,
)
VALIDATE_PATTERN = re.compile(
    r"\b("
    r"validate|"
    r"is this (yaml|manifest|valid|a valid)|"
    r"check this (yaml|manifest)|"
    r"is the (yaml|manifest) (valid|correct)|"
    r"verify (this )?(yaml|manifest)|"
    r"check (the |this )?(yaml|manifest|syntax)"
    r")\b",
    re.IGNORECASE,
)
MATH_KEYWORDS_PATTERN = re.compile(
    r"\b("
    r"calculate|compute|how many|how much|total memory|total cpu|"
    r"fit on|fits on|will fit|can I run|can fit"
    r")\b",
    re.IGNORECASE,
)
K8S_UNITS_PATTERN = re.compile(
    r"\d+\s*(GB|GiB|MB|MiB|Mi|Gi|Ki|cores?|millicore|m\b|"
    r"nodes?|pods?|replicas?|containers?)",
    re.IGNORECASE,
)


# ---------- LLM classifier prompt ----------

CLASSIFIER_SYSTEM = (
    "You are a strict query classifier. You output exactly ONE word: "
    "'casual', 'direct', or 'tools'. No punctuation, no explanation."
)

CLASSIFIER_USER_TEMPLATE = """Classify this Kubernetes-assistant query.

casual  = greetings, thanks, small talk, off-topic, questions about the assistant
direct  = a K8s question answerable directly (commands, concepts, definitions)
tools   = needs validation, code execution, math, or multi-step reasoning

Examples:
"hi" -> casual
"thanks!" -> casual
"what can you do" -> casual
"what does kubectl rollout undo do" -> direct
"how do I scale a deployment" -> direct
"explain Deployment vs StatefulSet" -> direct
"validate this YAML: apiVersion: v1..." -> tools
"calculate how many pods fit on 5 nodes 16GB each" -> tools
"check this manifest and tell me total memory" -> tools

Query: {query}

Label:"""

VALID_LABELS = {"casual", "direct", "tools"}


# ---------- Router ----------

class Router:
    def __init__(
        self,
        inference_url: str = DEFAULT_INFERENCE_URL,
        model: str = DEFAULT_MODEL,
    ):
        self.inference_url = inference_url
        self.model = model
        logger.info(f"Router initialized. inference_url={inference_url} model={model}")

    def keyword_route(self, query: str) -> str | None:
        if not query or not query.strip():
            return "casual"
        if GREETING_PATTERN.search(query):
            return "casual"
        if ASSISTANT_QUESTION_PATTERN.search(query):
            return "casual"
        if YAML_FIELDS_PATTERN.search(query):
            logger.info("Keyword route: YAML block detected -> tools")
            return "tools"
        if VALIDATE_PATTERN.search(query):
            logger.info("Keyword route: validation request -> tools")
            return "tools"
        if MATH_KEYWORDS_PATTERN.search(query) and K8S_UNITS_PATTERN.search(query):
            logger.info("Keyword route: math+units -> tools")
            return "tools"
        return None

    async def llm_classify(self, query: str) -> str:
        payload = {
            "messages": [
                {"role": "system", "content": CLASSIFIER_SYSTEM},
                {
                    "role": "user",
                    "content": CLASSIFIER_USER_TEMPLATE.format(query=query),
                },
            ],
            "model": self.model,
            "temperature": 0.0,
            "max_tokens": 5,
            "stop": ["\n", ".", " "],
        }

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                r = await client.post(f"{self.inference_url}/chat", json=payload)
                r.raise_for_status()
        except Exception as e:
            logger.error(f"Router HTTP error: {e}. Defaulting to 'direct'.")
            return "direct"

        raw = r.json().get("response", "").strip().lower()
        raw = raw.lstrip("`'\"-:* \t").rstrip("`'\".,;:* \t")

        if raw in VALID_LABELS:
            return raw
        for label in VALID_LABELS:
            if label in raw:
                return label

        logger.warning(
            f"Router could not parse '{raw}'. Defaulting to 'direct'."
        )
        return "direct"

    async def classify(self, query: str) -> str:
        kw = self.keyword_route(query)
        if kw is not None:
            logger.info(f"Router (keyword): '{query[:60]}' -> {kw}")
            return kw
        label = await self.llm_classify(query)
        logger.info(f"Router (LLM): '{query[:60]}' -> {label}")
        return label
