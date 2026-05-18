
import json
import random
import re
from pathlib import Path

from bs4 import BeautifulSoup
from datasets import load_dataset

# Fixed seed for reproducibility
random.seed(42)

# Output paths
OUTPUT_DIR = Path("./data")
OUTPUT_DIR.mkdir(exist_ok=True)

# Filter thresholds (tweak if needed)
MIN_QUESTION_LEN = 50
MAX_QUESTION_LEN = 2000
MIN_ANSWER_LEN = 100
MAX_ANSWER_LEN = 1500
MAX_QUESTION_MARKS = 2  # Reject multi-part questions
TARGET_SAMPLES = 300

# Train/val/test split (80/10/10 of TARGET_SAMPLES)
TRAIN_SIZE = 240
VAL_SIZE = 30
TEST_SIZE = 30


def html_to_markdown(html: str) -> str:
    """Strip HTML tags, preserve code blocks as markdown."""
    if not html:
        return ""

    soup = BeautifulSoup(html, "html.parser")

    # Convert <pre><code> blocks to triple-backtick markdown
    for pre in soup.find_all("pre"):
        code = pre.find("code")
        text = code.get_text() if code else pre.get_text()
        pre.replace_with(f"\n```\n{text}\n```\n")

    # Inline <code> becomes backticks
    for code in soup.find_all("code"):
        code.replace_with(f"`{code.get_text()}`")

    # Strip remaining tags, collapse whitespace
    text = soup.get_text()
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return text.strip()


def is_quality_sample(question: str, answer: str) -> bool:
    """Apply quality filters. Returns True if sample should be kept."""
    if not (MIN_QUESTION_LEN <= len(question) <= MAX_QUESTION_LEN):
        return False
    if not (MIN_ANSWER_LEN <= len(answer) <= MAX_ANSWER_LEN):
        return False

    # Multi-part questions confuse training
    if question.count("?") > MAX_QUESTION_MARKS:
        return False

    # Drop "answer is just a list of links"
    if answer.count("http") > 3 and len(answer) < 300:
        return False

    # Drop messy edit-heavy answers
    if any(marker in answer for marker in ["EDIT:", "UPDATE:", "EDIT2:"]):
        return False
    if question.count("EDIT") > 1 or question.count("UPDATE") > 1:
        return False

    return True


def quality_score(question: str, answer: str) -> float:
    """Score samples for ranking. Higher = better."""
    score = 0.0

    # Sweet spot for answer length
    if 200 <= len(answer) <= 800:
        score += 1.0

    # Bonus for code blocks (concrete answers)
    if "```" in answer:
        score += 0.5

    # Bonus for inline code references
    score += min(answer.count("`"), 10) * 0.05

    # Penalty for very long questions (often noisy)
    if len(question) > 1000:
        score -= 0.3

    return score


def to_chatml(question: str, answer: str) -> dict:
    """Convert to ChatML format expected by Phi-3-mini."""
    return {
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    }


def main():
    print("Loading dataset from HuggingFace...")
    ds = load_dataset("mcipriano/stackoverflow-kubernetes-questions", split="train")
    print(f"  Loaded {len(ds)} raw samples.\n")

    # Stage 1+2: Strip HTML
    print("Stripping HTML and converting to markdown...")
    cleaned = []
    for i, row in enumerate(ds):
        if i % 5000 == 0 and i > 0:
            print(f"  Processed {i}/{len(ds)}")
        question = html_to_markdown(row["Question"])
        answer = html_to_markdown(row["Answer"])
        if question and answer:
            cleaned.append({"question": question, "answer": answer})
    print(f"  After HTML cleaning: {len(cleaned)} samples.\n")

    # Stage 3: Quality filter
    print("Applying quality filters...")
    filtered = [
        s for s in cleaned if is_quality_sample(s["question"], s["answer"])
    ]
    print(f"  After quality filter: {len(filtered)} samples.\n")

    if len(filtered) < TARGET_SAMPLES:
        raise RuntimeError(
            f"Only {len(filtered)} samples passed filters, "
            f"need at least {TARGET_SAMPLES}. Loosen the filters."
        )

    # Stage 4: Score and pick top samples with diversity check
    print("Scoring and ranking...")
    for s in filtered:
        s["score"] = quality_score(s["question"], s["answer"])
    filtered.sort(key=lambda s: s["score"], reverse=True)

    print(f"Selecting top {TARGET_SAMPLES} with diversity check...")
    selected = []
    seen_starts = set()
    for s in filtered:
        # Diversity: first 80 chars of question must be unique
        q_start = s["question"][:80].lower()
        if q_start in seen_starts:
            continue
        seen_starts.add(q_start)
        selected.append(s)
        if len(selected) >= TARGET_SAMPLES:
            break

    print(f"  Selected {len(selected)} diverse samples.\n")

    # Stage 5: Convert to ChatML
    chatml_samples = [to_chatml(s["question"], s["answer"]) for s in selected]

    # Stage 6: Split 240/30/30
    print(f"Splitting into train/val/test ({TRAIN_SIZE}/{VAL_SIZE}/{TEST_SIZE})...")
    random.shuffle(chatml_samples)
    train = chatml_samples[:TRAIN_SIZE]
    val = chatml_samples[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE]
    test = chatml_samples[TRAIN_SIZE + VAL_SIZE:TRAIN_SIZE + VAL_SIZE + TEST_SIZE]

    for name, split in [("train", train), ("val", val), ("test", test)]:
        path = OUTPUT_DIR / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for sample in split:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        print(f"  Wrote {len(split)} samples to {path}")

    # Spot-check
    print("\n" + "=" * 60)
    print("SPOT CHECK: 3 random samples from final dataset")
    print("=" * 60)
    for i, s in enumerate(random.sample(chatml_samples, 3)):
        print(f"\n--- Sample {i + 1} ---")
        q = s["messages"][0]["content"]
        a = s["messages"][1]["content"]
        print(f"Q ({len(q)} chars): {q[:250]}{'...' if len(q) > 250 else ''}")
        print(f"A ({len(a)} chars): {a[:250]}{'...' if len(a) > 250 else ''}")


if __name__ == "__main__":
    main()
