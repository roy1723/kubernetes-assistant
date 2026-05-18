import subprocess
import sys
import tempfile
from pathlib import Path

import chromadb
import yaml
from sentence_transformers import SentenceTransformer

# Path resolution: this file is at mcp_server/tools.py; chroma is at data/chroma/
_PROJECT_ROOT = Path(__file__).parent.parent
_CHROMA_DIR = _PROJECT_ROOT / "data" / "chroma"
_COLLECTION_NAME = "kubernetes_docs"
_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Lazy globals so we don't load the embedding model unless search is called
_chroma_client = None
_collection = None
_embed_model = None


def _ensure_search_ready():
    """Lazy-initialize the embedding model and ChromaDB collection."""
    global _chroma_client, _collection, _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(_EMBEDDING_MODEL_NAME)
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=str(_CHROMA_DIR))
        _collection = _chroma_client.get_collection(_COLLECTION_NAME)


# ---------- Tool 1: search_documents ----------

def search_documents(query: str, top_k: int = 3) -> str:
    """
    Semantic search over the Kubernetes documentation vector store.
    Returns formatted text with top_k matches and source URLs.
    """
    _ensure_search_ready()

    query_emb = _embed_model.encode([query]).tolist()
    results = _collection.query(
        query_embeddings=query_emb,
        n_results=min(max(top_k, 1), 10),
    )

    docs = results["documents"][0] if results["documents"] else []
    metas = results["metadatas"][0] if results["metadatas"] else []

    if not docs:
        return "No relevant documents found."

    lines = [f"Found {len(docs)} relevant document(s):\n"]
    for i, (doc, meta) in enumerate(zip(docs, metas, strict=False), 1):
        lines.append(f"--- Result {i} ---")
        lines.append(f"Title: {meta['title']} > {meta['section']}")
        lines.append(f"Source: {meta['url']}")
        snippet = doc[:800] + ("..." if len(doc) > 800 else "")
        lines.append(f"Content: {snippet}")
        lines.append("")

    return "\n".join(lines)


# ---------- Tool 2: run_python ----------

def run_python(code: str, timeout: int = 10) -> str:
    """
    Execute Python code in a sandboxed subprocess.
    Captures stdout, stderr, and exit code. Times out after `timeout` seconds.
    """
    if not code or not code.strip():
        return "Error: empty code provided."

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        parts = []
        if result.stdout:
            parts.append("--- stdout ---\n" + result.stdout.strip())
        if result.stderr:
            parts.append("--- stderr ---\n" + result.stderr.strip())
        parts.append(f"--- exit code: {result.returncode} ---")
        return "\n\n".join(parts) if parts else "(no output)"

    except subprocess.TimeoutExpired:
        return f"Error: code execution exceeded {timeout}s timeout."
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ---------- Tool 3: validate_yaml ----------

def validate_yaml(yaml_text: str) -> str:
    """
    Parse YAML and check that each document has the basics of a K8s resource:
    apiVersion, kind, metadata.name.
    """
    if not yaml_text or not yaml_text.strip():
        return "Error: empty YAML provided."

    try:
        documents = list(yaml.safe_load_all(yaml_text))
    except yaml.YAMLError as e:
        return f"YAML syntax error:\n{e}"

    if not documents:
        return "Warning: no YAML documents found in input."

    findings = []
    for i, doc in enumerate(documents, 1):
        if doc is None:
            continue
        if not isinstance(doc, dict):
            findings.append(
                f"Document {i}: not a mapping (got {type(doc).__name__})"
            )
            continue

        issues = []
        if "apiVersion" not in doc:
            issues.append("missing 'apiVersion'")
        if "kind" not in doc:
            issues.append("missing 'kind'")
        if "metadata" not in doc:
            issues.append("missing 'metadata'")
        elif not isinstance(doc.get("metadata"), dict):
            issues.append("'metadata' is not a mapping")
        elif "name" not in doc["metadata"]:
            issues.append("missing 'metadata.name'")

        kind = doc.get("kind", "Unknown")
        api_v = doc.get("apiVersion", "?")
        if issues:
            findings.append(
                f"Document {i} ({kind} @ {api_v}): {', '.join(issues)}"
            )
        else:
            findings.append(f"Document {i} ({kind} @ {api_v}): valid structure")

    n_valid = sum(1 for line in findings if line.endswith("valid structure"))
    summary = (
        f"YAML parsed successfully. {len(documents)} document(s) found, "
        f"{n_valid} pass basic K8s validation."
    )
    return summary + "\n\n" + "\n".join(findings)
