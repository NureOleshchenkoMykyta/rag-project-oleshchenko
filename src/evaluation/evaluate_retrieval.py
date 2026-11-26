from __future__ import annotations

import argparse
import json
from pathlib import Path

import chromadb
import yaml
from sentence_transformers import SentenceTransformer


def load_yaml_config(config_path: Path | None) -> dict:
    if config_path and config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def load_eval_set(eval_path: Path) -> list[dict]:
    items: list[dict] = []
    with eval_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top_k_set = set(retrieved_ids[:k])
    hit_count = len(top_k_set & relevant_ids)
    return hit_count / len(relevant_ids)


def evaluate_run(
    eval_items: list[dict],
    model: SentenceTransformer,
    collection,
    k_max: int = 10,
    where_filter: dict | None = None,
) -> tuple[float, float, list[dict]]:
    queries = [item["query"] for item in eval_items]
    query_vecs = model.encode(
        queries,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).tolist()

    query_kwargs = {
        "query_embeddings": query_vecs,
        "n_results": k_max,
        "include": ["metadatas", "distances"],
    }
    if where_filter is not None:
        query_kwargs["where"] = where_filter

    res = collection.query(**query_kwargs)
    all_ids = res.get("ids", [])

    r5_scores: list[float] = []
    r10_scores: list[float] = []
    details: list[dict] = []

    for item, retrieved in zip(eval_items, all_ids):
        relevant = set(item["relevant_chunk_ids"])
        r5 = recall_at_k(retrieved, relevant, 5)
        r10 = recall_at_k(retrieved, relevant, 10)
        r5_scores.append(r5)
        r10_scores.append(r10)
        details.append(
            {
                "query_id": item["query_id"],
                "category": item.get("category", ""),
                "relevant": sorted(relevant),
                "top5": retrieved[:5],
                "top10": retrieved[:10],
                "recall@5": r5,
                "recall@10": r10,
            }
        )

    mean_r5 = sum(r5_scores) / len(r5_scores) if r5_scores else 0.0
    mean_r10 = sum(r10_scores) / len(r10_scores) if r10_scores else 0.0
    return mean_r5, mean_r10, details


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Recall@5 and Recall@10 on eval-set.jsonl.")
    parser.add_argument(
        "--eval-set",
        type=Path,
        default=Path("tests/eval-set.jsonl"),
        help="Path to eval-set.jsonl",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/settings.yaml"),
        help="Path to settings.yaml",
    )
    parser.add_argument(
        "--show-misses",
        action="store_true",
        help="Print queries where Recall@5 < 1.0 or Recall@10 < 1.0",
    )
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    emb_cfg = cfg.get("embeddings", {})
    vs_cfg = cfg.get("vector_store", {})

    model_name = emb_cfg.get("model_name", "all-MiniLM-L6-v2")
    persist_dir = Path(vs_cfg.get("persist_directory", "data/chroma_db"))
    collection_name = vs_cfg.get("collection_name", "se_stackexchange")

    eval_items = load_eval_set(args.eval_set)
    model = SentenceTransformer(model_name)
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_collection(name=collection_name)

    base_r5, base_r10, base_details = evaluate_run(eval_items, model, collection, k_max=10, where_filter=None)

    c0_r5, c0_r10, _ = evaluate_run(
        eval_items, model, collection, k_max=10, where_filter={"chunk_index": 0}
    )

    print("\n=== RETRIEVAL EVALUATION SUMMARY ===")
    print(f"Eval queries count       : {len(eval_items)}")
    print(f"[Baseline] Recall@5      : {base_r5:.4f} ({base_r5 * 100:.1f}%)")
    print(f"[Baseline] Recall@10     : {base_r10:.4f} ({base_r10 * 100:.1f}%)")
    print(f"[chunk_index==0] Recall@5: {c0_r5:.4f} ({c0_r5 * 100:.1f}%)")
    print(f"[chunk_index==0] Recall@10: {c0_r10:.4f} ({c0_r10 * 100:.1f}%)")

    if args.show_misses:
        print("\n=== QUERIES WITH RECALL < 1.0 (BASELINE) ===")
        for d in base_details:
            if d["recall@5"] < 1.0 or d["recall@10"] < 1.0:
                print(
                    f"{d['query_id']} ({d['category']}): R@5={d['recall@5']:.2f}, R@10={d['recall@10']:.2f} | "
                    f"rel={d['relevant']} | top5={d['top5']}"
                )


if __name__ == "__main__":
    main()