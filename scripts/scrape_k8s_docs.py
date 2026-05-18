import json
import random
import re
import time
from pathlib import Path

import requests

# Curated K8s concept docs. (display_title, github_path_under_concepts/)
DOCS = [
    ("Containers", "containers/_index.md"),
    ("Container Images", "containers/images.md"),
    ("Pods", "workloads/pods/_index.md"),
    ("Pod Lifecycle", "workloads/pods/pod-lifecycle.md"),
    ("Deployments", "workloads/controllers/deployment.md"),
    ("ReplicaSets", "workloads/controllers/replicaset.md"),
    ("StatefulSets", "workloads/controllers/statefulset.md"),
    ("DaemonSets", "workloads/controllers/daemonset.md"),
    ("Jobs", "workloads/controllers/job.md"),
    ("CronJobs", "workloads/controllers/cron-jobs.md"),
    ("Services", "services-networking/service.md"),
    ("Ingress", "services-networking/ingress.md"),
    ("NetworkPolicies", "services-networking/network-policies.md"),
    ("DNS for Services and Pods", "services-networking/dns-pod-service.md"),
    ("Volumes", "storage/volumes.md"),
    ("Persistent Volumes", "storage/persistent-volumes.md"),
    ("Storage Classes", "storage/storage-classes.md"),
    ("ConfigMaps", "configuration/configmap.md"),
    ("Secrets", "configuration/secret.md"),
    ("Resource Management for Pods and Containers", "configuration/manage-resources-containers.md"),
    ("Namespaces", "overview/working-with-objects/namespaces.md"),
    ("Labels and Selectors", "overview/working-with-objects/labels.md"),
    ("Annotations", "overview/working-with-objects/annotations.md"),
    ("Kubernetes Objects", "overview/working-with-objects/_index.md"),
]

BASE_URL = "https://raw.githubusercontent.com/kubernetes/website/main/content/en/docs/concepts"
WEB_BASE = "https://kubernetes.io/docs/concepts"

OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "k8s_docs.json"


def strip_frontmatter(text: str) -> str:
    """Remove the YAML frontmatter block (--- ... ---) at the start."""
    if text.startswith("---"):
        match = re.match(r"^---\s*\n.*?\n---\s*\n", text, re.DOTALL)
        if match:
            return text[match.end():]
    return text


def strip_hugo_shortcodes(text: str) -> str:
    """Remove Hugo template tags like {{< caution >}}, {{% codenew %}}, etc."""
    text = re.sub(r"\{\{[<%].*?[%>]\}\}", "", text, flags=re.DOTALL)
    text = re.sub(r"\n\s*\n\s*\n", "\n\n", text)
    return text


def chunk_by_section(content: str, title: str, url: str) -> list[dict]:
    """Split content by H2 headers into chunks. Skip very short sections."""
    parts = re.split(r"\n(?=## )", content)
    chunks = []

    # Handle intro paragraph (everything before first H2)
    if parts and not parts[0].startswith("## "):
        intro = parts[0].strip()
        intro = re.sub(r"^# .*\n", "", intro, count=1)  # remove H1
        if len(intro) > 200:
            chunks.append({
                "title": title,
                "section": "Introduction",
                "content": intro,
                "url": url,
            })
        parts = parts[1:]

    for part in parts:
        if not part.strip():
            continue
        first_line, *rest = part.split("\n", 1)
        section_name = first_line.lstrip("# ").strip()
        body = rest[0].strip() if rest else ""

        if len(body) < 200:
            continue
        if len(body) > 2000:
            body = body[:2000].rsplit("\n", 1)[0] + "..."

        chunks.append({
            "title": title,
            "section": section_name,
            "content": body,
            "url": url,
        })

    return chunks


def main():
    all_chunks = []

    for i, (title, path) in enumerate(DOCS, 1):
        full_url = f"{BASE_URL}/{path}"
        web_url = f"{WEB_BASE}/{path.replace('/_index.md', '/').replace('.md', '/')}"

        print(f"[{i}/{len(DOCS)}] Fetching {title}...")
        try:
            resp = requests.get(full_url, timeout=15)
            resp.raise_for_status()
        except Exception as e:
            print(f"  FAILED: {e}")
            continue

        text = strip_frontmatter(resp.text)
        text = strip_hugo_shortcodes(text)

        chunks = chunk_by_section(text, title, web_url)
        print(f"  -> {len(chunks)} chunks")

        for chunk in chunks:
            slug = re.sub(r"[^a-z0-9]+", "_", chunk["section"].lower()).strip("_")
            base_id = path.replace("/", "_").replace(".md", "")
            chunk["id"] = f"{base_id}__{slug}"[:180]
            all_chunks.append(chunk)

        time.sleep(0.4)  # be polite to GitHub

    # Dedupe IDs if any collisions
    seen = {}
    for c in all_chunks:
        if c["id"] in seen:
            seen[c["id"]] += 1
            c["id"] = f"{c['id']}_{seen[c['id']]}"
        else:
            seen[c["id"]] = 0

    print(f"\nTotal chunks: {len(all_chunks)}")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    print(f"Saved to {OUTPUT_FILE}")

    # Spot-check
    print("\n--- 3 random sample chunks ---")
    for c in random.sample(all_chunks, min(3, len(all_chunks))):
        print(f"\n[{c['title']}] {c['section']}")
        print(f"URL: {c['url']}")
        print(f"Content preview: {c['content'][:200]}...")


if __name__ == "__main__":
    main()
