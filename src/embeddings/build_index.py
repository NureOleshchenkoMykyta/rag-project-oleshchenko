from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import chromadb
import yaml
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def load_yaml_config(config_path: Path | None) -> dict:
    if config_path and config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def load_chunks(chunks_path: Path) -> list[dict]:
    chunks: list[dict] = []
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def get_dir_size_bytes(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Encode chunks with SentenceTransformer and build persistent ChromaDB index."
    )
    parser.add_argument(
        "--chunks-path",
        type=Path,
        default=Path("data/processed/chunks.jsonl"),
        help="Path to processed chunks JSONL file",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/settings.yaml"),
        help="Path to settings.yaml",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for embedding inference and Chroma upsert",
    )
    parser.add_argument(
        "--space",
        type=str,
        default="cosine",
        choices=["cosine", "l2", "ip"],
        help="HNSW distance metric in ChromaDB",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        default=True,
        help="Recreate persist directory before indexing",
    )
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    emb_cfg = cfg.get("embeddings", {})
    vs_cfg = cfg.get("vector_store", {})

    model_name = emb_cfg.get("model_name", "all-MiniLM-L6-v2")
    persist_dir = Path(vs_cfg.get("persist_directory", "data/chroma_db"))
    collection_name = vs_cfg.get("collection_name", "se_stackexchange")

    if not args.chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {args.chunks_path}")

    chunks = load_chunks(args.chunks_path)
    print(f"Loaded {len(chunks)} chunks from {args.chunks_path}")

    t0 = time.perf_counter()
    print(f"Loading embedding model: {model_name} ...")
    model = SentenceTransformer(model_name)

    texts = [c["text"] for c in chunks]
    print("Encoding chunks into dense vectors ...")
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    encode_sec = time.perf_counter() - t0
    dim = int(embeddings.shape[1]) if len(embeddings) > 0 else 0

    if args.reset and persist_dir.exists():
        shutil.rmtree(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": args.space},
    )

    t1 = time.perf_counter()
    total = len(chunks)
    for start in tqdm(range(0, total, args.batch_size), desc="Indexing into ChromaDB"):
        end = min(start + args.batch_size, total)
        batch_chunks = chunks[start:end]
        batch_embs = embeddings[start:end].tolist()

        ids = [c["chunk_id"] for c in batch_chunks]
        docs = [c["text"] for c in batch_chunks]
        metadatas = [
            {
                "chunk_id": str(c["chunk_id"]),
                "doc_id": str(c["doc_id"]),
                "chunk_index": int(c["chunk_index"]),
                "total_chunks": int(c["total_chunks"]),
                "title": str(c.get("title", "")),
                "source": str(c.get("source", "")),
                "license": str(c.get("license", "")),
            }
            for c in batch_chunks
        ]

        collection.add(
            ids=ids,
            embeddings=batch_embs,
            documents=docs,
            metadatas=metadatas,
        )

    index_sec = time.perf_counter() - t1
    disk_bytes = get_dir_size_bytes(persist_dir)

    print("\n=== VECTOR INDEX SUMMARY ===")
    print(f"Embedding model          : {model_name}")
    print(f"Vector dimension         : {dim}")
    print(f"Distance metric          : {args.space}")
    print(f"Persist directory        : {persist_dir}")
    print(f"Collection name          : {collection_name}")
    print(f"Total vectors indexed    : {collection.count()}")
    print(f"Encoding time            : {encode_sec:.2f} s")
    print(f"ChromaDB indexing time   : {index_sec:.2f} s")
    print(f"Index size on disk       : {disk_bytes:,} bytes ({disk_bytes / (1024 * 1024):.2f} MB)")


if __name__ == "__main__":
    main()