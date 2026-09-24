from __future__ import annotations

from ..models import SearchResult
from .tokenizer import tokenize


class Reranker:
    name = "base"
    def score(self, query: str, documents: list[str]) -> list[float]: raise NotImplementedError


class OverlapReranker(Reranker):
    name = "overlap"
    def score(self, query, documents):
        q = set(tokenize(query)); return [len(q & set(tokenize(d))) / max(len(q), 1) for d in documents]


class CrossEncoderReranker(Reranker):
    def __init__(self, model_name: str, *, local_files_only: bool = False):
        try:
            from sentence_transformers import CrossEncoder
            import torch
        except ImportError as exc:
            raise RuntimeError("sentence-transformers is required for Cross-Encoder reranking; install '.[ml]'") from exc
        self.model = CrossEncoder(model_name, local_files_only=local_files_only, activation_fn=torch.nn.Sigmoid())
        self.name = model_name
    def score(self, query, documents):
        pairs = [(query, doc) for doc in documents]
        return [float(x) for x in self.model.predict(pairs, show_progress_bar=False, batch_size=32)]


def create_reranker(name: str) -> Reranker:
    if name == "overlap": return OverlapReranker()
    return CrossEncoderReranker(name)


def rerank(query, results, reranker, top_k):
    if not results: return []
    scores = reranker.score(query, [r.text for r in results])
    rows = [SearchResult(r.chunk_id, float(s), "reranked", r.text, r.metadata) for r, s in zip(results, scores)]
    return sorted(rows, key=lambda x: x.score, reverse=True)[:top_k]
