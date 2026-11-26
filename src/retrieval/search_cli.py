from __future__ import annotations

import argparse
from pathlib import Path

import chromadb
import yaml
from sentence_transformers import SentenceTransformer


def load_yaml_config(config_path: Path | None) -> dict:
    if config_path and config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def search_top_k(
    query: str,
    model: SentenceTransformer,
    collection,
    top_k: int = 5,
) -> list[dict]:
    query_vec = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).tolist()

    res = collection.query(
        query_embeddings=query_vec,
        n_results=top_k,
        include=["metadatas", "documents", "distances"],
    )

    hits: list[dict] = []
    ids = res.get("ids", [[]])[0]
    dists = res.get("distances", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    docs = res.get("documents", [[]])[0]

    for rank, (cid, dist, meta, doc) in enumerate(zip(ids, dists, metas, docs), start=1):
        similarity = 1.0 - float(dist)
        hits.append(
            {
                "rank": rank,
                "chunk_id": cid,
                "similarity": similarity,
                "distance": float(dist),
                "title": (meta or {}).get("title", ""),
                "doc_id": (meta or {}).get("doc_id", ""),
                "preview": " ".join((doc or "").split())[:220],
            }
        )
    return hits


def main() -> None:
    parser = argparse.ArgumentParser(description="Search top-K chunks in ChromaDB from CLI.")
    parser.add_argument("query", type=str, help="Search query text")
    parser.add_argument("--top-k", type=int, default=None, help="Number of results to return")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/settings.yaml"),
        help="Path to settings.yaml",
    )
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    emb_cfg = cfg.get("embeddings", {})
    vs_cfg = cfg.get("vector_store", {})
    ret_cfg = cfg.get("retrieval", {})

    model_name = emb_cfg.get("model_name", "all-MiniLM-L6-v2")
    persist_dir = Path(vs_cfg.get("persist_directory", "data/chroma_db"))
    collection_name = vs_cfg.get("collection_name", "se_stackexchange")
    top_k = args.top_k if args.top_k is not None else int(ret_cfg.get("top_k", 5))

    model = SentenceTransformer(model_name)
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_collection(name=collection_name)

    hits = search_top_k(args.query, model=model, collection=collection, top_k=top_k)

    print(f'\nQuery: "{args.query}" (top-{top_k})\n')
    for h in hits:
        print(
            f"[{h['rank']}] chunk_id={h['chunk_id']} | sim={h['similarity']:.4f} "
            f"(dist={h['distance']:.4f}) | title={h['title']}"
        )
        print(f"    {h['preview']}...")


if __name__ == "__main__":
    main()