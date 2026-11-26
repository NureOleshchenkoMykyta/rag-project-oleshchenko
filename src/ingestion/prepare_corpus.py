from __future__ import annotations

import argparse
import collections
import html
import json
import random
import re
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml
from bs4 import BeautifulSoup
from tqdm import tqdm


STOPWORDS = {
    "the", "and", "to", "of", "a", "in", "is", "for", "that", "it", "on", "with",
    "as", "are", "be", "this", "or", "an", "by", "from", "have", "not", "can",
    "if", "you", "your", "we", "they", "which", "but", "all", "one", "more",
    "so", "what", "when", "where", "how", "who", "will", "would", "could",
    "should", "has", "had", "was", "were", "been", "do", "does", "did", "use",
    "used", "using", "into", "out", "up", "at", "about", "there", "their", "them",
    "some", "any", "no", "just", "like", "also", "only", "other", "than", "then",
    "very", "each", "over", "after", "before", "between", "through", "during",
    "without", "within", "along", "following", "across", "behind", "beyond",
    "plus", "except", "but", "up", "down", "off", "over", "under", "again",
    "further", "once", "here", "why", "both", "few", "most", "own", "same",
    "too", "s", "t", "don", "re", "ve", "ll", "m", "d", "i", "me", "my", "he",
    "him", "his", "she", "her", "its", "our", "ours", "you", "your", "yours",
}


def load_yaml_config(config_path: Path | None) -> dict:
    if config_path and config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def clean_html_text(raw_html: str) -> str:
    if not raw_html:
        return ""
    text = unicodedata.normalize("NFC", raw_html)
    text = html.unescape(text)
    soup = BeautifulSoup(text, "html.parser")
    plain = soup.get_text(separator=" ")
    plain = html.unescape(plain)
    plain = unicodedata.normalize("NFC", plain)
    plain = re.sub(r"[ \t\r\f\v]+", " ", plain)
    plain = re.sub(r"\n\s*\n+", "\n\n", plain)
    return plain.strip()


def chunk_words(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_size:
        return [" ".join(words)]

    step = max(1, chunk_size - overlap)
    chunks: list[str] = []
    for start in range(0, len(words), step):
        part = words[start : start + chunk_size]
        if not part:
            continue
        chunks.append(" ".join(part))
        if start + chunk_size >= len(words):
            break
    return chunks


def extract_documents_from_posts_xml(
    xml_path: Path,
    max_docs: int,
    min_chars: int = 100,
) -> tuple[int, list[dict]]:
    """
    Читает Posts.xml потоково, собирает вопросы (PostTypeId=1) с принятым
    или лучшим ответом (PostTypeId=2) и возвращает (raw_candidates_count, cleaned_docs).
    """
    questions: dict[str, dict] = {}
    answers_by_parent: dict[str, list[dict]] = collections.defaultdict(list)
    answers_by_id: dict[str, dict] = {}

    context = ET.iterparse(str(xml_path), events=("end",))
    for _, elem in context:
        if elem.tag != "row":
            continue
        attrs = elem.attrib
        post_type = attrs.get("PostTypeId")
        post_id = attrs.get("Id")
        if post_type == "1" and post_id:
            questions[post_id] = {
                "id": post_id,
                "title": attrs.get("Title", ""),
                "body": attrs.get("Body", ""),
                "accepted_answer_id": attrs.get("AcceptedAnswerId"),
                "score": int(attrs.get("Score", "0") or "0"),
            }
        elif post_type == "2" and post_id:
            parent_id = attrs.get("ParentId")
            ans = {
                "id": post_id,
                "parent_id": parent_id,
                "body": attrs.get("Body", ""),
                "score": int(attrs.get("Score", "0") or "0"),
            }
            answers_by_id[post_id] = ans
            if parent_id:
                answers_by_parent[parent_id].append(ans)

        elem.clear()

    raw_candidates_count = len(questions)
    cleaned_docs: list[dict] = []

    for q_id, q in questions.items():
        if len(cleaned_docs) >= max_docs:
            break

        chosen_answer = None
        acc_id = q.get("accepted_answer_id")
        if acc_id and acc_id in answers_by_id:
            chosen_answer = answers_by_id[acc_id]
        elif q_id in answers_by_parent and answers_by_parent[q_id]:
            chosen_answer = max(answers_by_parent[q_id], key=lambda x: x["score"])

        title_clean = clean_html_text(q["title"])
        q_body_clean = clean_html_text(q["body"])
        a_body_clean = clean_html_text(chosen_answer["body"]) if chosen_answer else ""

        parts = []
        if title_clean:
            parts.append(f"Question: {title_clean}")
        if q_body_clean:
            parts.append(q_body_clean)
        if a_body_clean:
            parts.append(f"Accepted/Top Answer:\n{a_body_clean}")

        full_text = "\n\n".join(parts).strip()
        if len(full_text) < min_chars:
            continue

        cleaned_docs.append(
            {
                "doc_id": f"se_{q_id}",
                "title": title_clean,
                "text": full_text,
                "source": "Software Engineering Stack Exchange (Posts.xml)",
                "license": "CC BY-SA 4.0",
            }
        )

    return raw_candidates_count, cleaned_docs


def compute_top_words(chunks: list[dict], top_n: int = 20) -> list[tuple[str, int]]:
    counter: collections.Counter[str] = collections.Counter()
    token_re = re.compile(r"[a-z]{3,}")
    for item in chunks:
        tokens = token_re.findall(item["text"].lower())
        filtered = [tok for tok in tokens if tok not in STOPWORDS]
        counter.update(filtered)
    return counter.most_common(top_n)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SE Stack Exchange corpus into JSONL chunks.")
    parser.add_argument(
        "--xml-path",
        type=Path,
        default=Path(r"D:\Datasets\se-dump\Posts.xml"),
        help="Path to Posts.xml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/chunks.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/settings.yaml"),
        help="Path to settings.yaml",
    )
    parser.add_argument("--max-docs", type=int, default=3000, help="Max cleaned documents (500..10000)")
    parser.add_argument("--min-chars", type=int, default=100, help="Minimum document length after cleaning")
    parser.add_argument("--chunk-size", type=int, default=None, help="Chunk size in words")
    parser.add_argument("--overlap", type=int, default=None, help="Overlap size in words")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sample inspection")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    chunk_size = args.chunk_size or int(cfg.get("chunk_size", 300))
    overlap = args.overlap or int(cfg.get("chunk_overlap", 50))

    raw_count, cleaned_docs = extract_documents_from_posts_xml(
        xml_path=args.xml_path,
        max_docs=args.max_docs,
        min_chars=args.min_chars,
    )

    all_chunks: list[dict] = []
    for doc in tqdm(cleaned_docs, desc="Chunking documents"):
        parts = chunk_words(doc["text"], chunk_size=chunk_size, overlap=overlap)
        total = len(parts)
        for idx, part_text in enumerate(parts):
            all_chunks.append(
                {
                    "chunk_id": f"{doc['doc_id']}_c{idx}",
                    "doc_id": doc["doc_id"],
                    "chunk_index": idx,
                    "total_chunks": total,
                    "title": doc["title"],
                    "text": part_text,
                    "source": doc["source"],
                    "license": doc["license"],
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in all_chunks:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    word_counts = [len(c["text"].split()) for c in all_chunks]
    char_counts = [len(c["text"]) for c in all_chunks]
    avg_words = sum(word_counts) / len(word_counts) if word_counts else 0.0
    avg_chars = sum(char_counts) / len(char_counts) if char_counts else 0.0
    top_words = compute_top_words(all_chunks, top_n=20)

    print("\n=== CORPUS PREPARATION SUMMARY ===")
    print(f"Question candidates in XML : {raw_count}")
    print(f"Cleaned documents saved    : {len(cleaned_docs)}")
    print(f"Total chunks generated     : {len(all_chunks)}")
    print(f"Average chunk length       : {avg_words:.1f} words ({avg_chars:.1f} chars)")
    print(f"Output JSONL size (bytes)  : {args.output.stat().st_size}")

    print("\n=== TOP 20 WORDS ===")
    for rank, (word, freq) in enumerate(top_words, start=1):
        print(f"{rank:2d}. {word}: {freq}")

    sample_size = min(10, len(cleaned_docs))
    rng = random.Random(args.seed)
    samples = rng.sample(cleaned_docs, k=sample_size)
    print(f"\n=== {sample_size} RANDOM CLEANED DOCUMENTS (MANUAL CHECK) ===")
    for i, doc in enumerate(samples, start=1):
        preview = doc["text"][:260].replace("\n", " ")
        print(f"[{i}] {doc['doc_id']} | len={len(doc['text'])} | {preview}...")


if __name__ == "__main__":
    main()