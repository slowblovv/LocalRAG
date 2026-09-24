from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from localrag.config import settings
from localrag.evaluation.metrics import ndcg_at_k, recall_at_k
from localrag.index import LocalIndex
from localrag.retrieval.hybrid import reciprocal_rank_fusion
from localrag.retrieval.rerank import OverlapReranker, create_reranker, rerank


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    return statistics.quantiles(values, n=20, method="inclusive")[-1] if len(values) >= 2 else values[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deterministic", action="store_true", help="use hash/overlap test doubles; never use this for the main portfolio benchmark")
    args = parser.parse_args()
    idx = LocalIndex()
    idx.build("demo_corpus")
    idx.load_current()
    if idx.embedder.name.startswith("hash-") and not args.deterministic:
        raise RuntimeError("main ablation must use the configured real SentenceTransformer embedder")
    qs = json.loads(Path("evaluation/queries.json").read_text())
    rel = json.loads(Path("evaluation/relevance.json").read_text())

    cross_encoder = create_reranker(settings.reranker_model)
    if cross_encoder.name == "overlap" and not args.deterministic:
        raise RuntimeError("main ablation must use a real Cross-Encoder reranker")

    configs = {
        "BM25": lambda q: idx.lexical.search(q, 40),
        "Dense": lambda q: idx.vector.search(q, 40, idx.chunks),
        "Hybrid RRF": lambda q: reciprocal_rank_fusion([idx.lexical.search(q, 40), idx.vector.search(q, 40, idx.chunks)], settings.rrf_k, 40),
    }
    rows: list[tuple[str, float, float, float, float]] = []
    for name, fn in configs.items():
        r10, n10, times = [], [], []
        for q, r in zip(qs, rel):
            t = time.perf_counter(); results = fn(q["query"]); times.append((time.perf_counter() - t) * 1000)
            ids = [x.chunk_id for x in results]
            r10.append(recall_at_k(ids, r["relevant_chunk_ids"], 10)); n10.append(ndcg_at_k(ids, r["relevant_chunk_ids"], 10))
        rows.append((name, sum(r10) / len(r10), sum(n10) / len(n10), statistics.mean(times), p95(times)))

    def measure_reranker(label: str, reranker) -> None:
        r10, n10, times = [], [], []
        for q, r in zip(qs, rel):
            t = time.perf_counter()
            base = configs["Hybrid RRF"](q["query"])
            results = rerank(q["query"], base, reranker, 10)
            times.append((time.perf_counter() - t) * 1000)
            ids = [x.chunk_id for x in results]
            r10.append(recall_at_k(ids, r["relevant_chunk_ids"], 10)); n10.append(ndcg_at_k(ids, r["relevant_chunk_ids"], 10))
        rows.append((label, sum(r10) / len(r10), sum(n10) / len(n10), statistics.mean(times), p95(times)))

    measure_reranker("Hybrid + Overlap baseline", OverlapReranker())
    measure_reranker("Hybrid + Cross-Encoder", cross_encoder)

    print("Pipeline | Recall@10 | NDCG@10 | mean latency ms | p95 latency ms")
    for name, recall, ndcg, mean_ms, p95_ms in rows:
        print(f"{name} | {recall:.4f} | {ndcg:.4f} | {mean_ms:.3f} | {p95_ms:.3f}")
    print({"embedding_model": idx.embedder.name, "reranker_model": cross_encoder.name, "rrf_k": settings.rrf_k})
    print("The comparison is measured from the same fixture corpus; numbers are never hard-coded.")


if __name__ == "__main__":
    main()
