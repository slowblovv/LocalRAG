from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from localrag.config import settings
from localrag.index import LocalIndex
from localrag.models import SearchResult
from localrag.retrieval.hybrid import reciprocal_rank_fusion
from localrag.retrieval.lexical import TinySearchAdapter
from localrag.retrieval.rerank import create_reranker, rerank
from localrag.retrieval.vector import VectorIndex


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[int(q) - 1]


def rss_mb() -> float:
    try:
        pages = int(Path("/proc/self/statm").read_text().split()[1])
        return pages * 4096 / (1024 * 1024)
    except (OSError, ValueError, IndexError):
        return 0.0


def scale_chunks(base_chunks, scale: int):
    out = []
    for i in range(scale):
        base = base_chunks[i % len(base_chunks)]
        out.append(replace(base, id=f"{base.id}-{i:08d}"))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="LocalRAG retrieval performance benchmark")
    parser.add_argument("--scales", nargs="+", type=int, default=[1_000, 10_000, 100_000])
    parser.add_argument("--embedding-sample", type=int, default=64)
    args = parser.parse_args()

    idx = LocalIndex()
    idx.build("demo_corpus")
    idx.load_current()
    if idx.embedder.name.startswith("hash-"):
        raise RuntimeError("performance benchmark requires the real configured SentenceTransformer embedder")
    reranker = create_reranker(settings.reranker_model)
    if reranker.name == "overlap":
        raise RuntimeError("performance benchmark requires the real configured Cross-Encoder reranker")

    sample = [c.text for c in list(idx.chunks.values())[:max(1, args.embedding_sample)]]
    t = time.perf_counter(); idx.embedder.embed_documents(sample); embedding_seconds = time.perf_counter() - t
    print(json.dumps({"embedding_model": idx.embedder.name, "reranker_model": reranker.name, "embedding_sample": len(sample), "embedding_throughput_chunks_per_sec": len(sample) / max(embedding_seconds, 1e-9)}, indent=2))

    query = "reliable ordered delivery and packet loss"
    base_chunks = list(idx.chunks.values())
    base_vectors = np.asarray(idx.vector.vectors)
    results = []
    for scale in sorted(set(args.scales)):
        chunks = scale_chunks(base_chunks, scale)
        lexical = TinySearchAdapter(); lexical.build(chunks)
        vector = VectorIndex(idx.embedder)
        repeats = int(np.ceil(scale / len(base_vectors)))
        vector.vectors = np.tile(base_vectors, (repeats, 1))[:scale].astype(np.float32, copy=False)
        vector.chunk_ids = [c.id for c in chunks]
        vector.index_version = f"benchmark-{scale}"

        bm_times, dense_times, hybrid_times, rerank_times = [], [], [], []
        # Warm-up each path once.
        lexical.search(query, 20); vector.search(query, 20, {c.id: c for c in chunks})
        for _ in range(25):
            t = time.perf_counter(); lex = lexical.search(query, 20); bm_times.append((time.perf_counter() - t) * 1000)
            t = time.perf_counter(); dense = vector.search(query, 20, {c.id: c for c in chunks}); dense_times.append((time.perf_counter() - t) * 1000)
            t = time.perf_counter(); hybrid = reciprocal_rank_fusion([lex, dense], settings.rrf_k, 40); hybrid_times.append((time.perf_counter() - t) * 1000)
            t = time.perf_counter(); rerank(query, hybrid, reranker, 10); rerank_times.append((time.perf_counter() - t) * 1000)

        with tempfile.TemporaryDirectory(prefix=f"localrag-benchmark-{scale}-") as td:
            tmp = Path(td)
            lexical.dump(tmp / "bm25.json")
            vector.save(tmp / "vector")
            disk_bytes = sum(p.stat().st_size for p in tmp.rglob("*" ) if p.is_file())

        results.append({
            "chunks": scale,
            "bm25_p50_ms": percentile(bm_times, 50), "bm25_p95_ms": percentile(bm_times, 95),
            "dense_p50_ms": percentile(dense_times, 50), "dense_p95_ms": percentile(dense_times, 95),
            "hybrid_p50_ms": percentile(hybrid_times, 50), "hybrid_p95_ms": percentile(hybrid_times, 95),
            "hybrid_reranker_p50_ms": percentile(rerank_times, 50), "hybrid_reranker_p95_ms": percentile(rerank_times, 95),
            "ram_mb": rss_mb(), "serialized_index_mb": disk_bytes / (1024 * 1024),
        })

    print(json.dumps({"results": results, "note": "Synthetic scaling reuses embeddings produced by the configured real SentenceTransformer model; benchmark results are measured, not claimed in advance."}, indent=2))


if __name__ == "__main__":
    main()
