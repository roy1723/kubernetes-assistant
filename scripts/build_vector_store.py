import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

CHUNKS_FILE = Path("data/k8s_docs.json")
CHROMA_DIR = Path("data/chroma")
COLLECTION_NAME = "kubernetes_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def main():
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading chunks from {CHUNKS_FILE}...")
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"  {len(chunks)} chunks loaded.\n")

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    print("  (downloads ~80 MB on first run)")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"  Embedding dim: {model.get_sentence_embedding_dimension()}\n")

    print(f"Setting up ChromaDB at {CHROMA_DIR}...")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Idempotent rebuild
    try:
        client.delete_collection(COLLECTION_NAME)
        print("  Deleted existing collection.")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    print(f"  Created collection '{COLLECTION_NAME}'.\n")

    print("Embedding chunks...")
    texts = [f"{c['title']}: {c['section']}\n\n{c['content']}" for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32).tolist()
    print("  Done.\n")

    print("Adding to ChromaDB...")
    collection.add(
        ids=[c["id"] for c in chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[
            {"title": c["title"], "section": c["section"], "url": c["url"]}
            for c in chunks
        ],
    )
    print(f"  Added {len(chunks)} documents.\n")

    # Verify
    print("Verifying with test query: 'how to roll back a deployment'")
    test_emb = model.encode(["how to roll back a deployment"]).tolist()
    results = collection.query(query_embeddings=test_emb, n_results=3)

    print("\nTop 3 results:")
    for i, (doc, meta) in enumerate(
        zip(results["documents"][0], results["metadatas"][0], strict=False), 1
    ):
        print(f"\n[{i}] {meta['title']} > {meta['section']}")
        print(f"    URL: {meta['url']}")
        print(f"    Content: {doc[:200]}...")

    print(f"\nVector store built. Collection size: {collection.count()}")


if __name__ == "__main__":
    main()
