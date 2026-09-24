from __future__ import annotations

import hashlib
import numpy as np


class Embedder:
    name = "base"; dimension = 0; normalization_policy = "l2"
    def embed_documents(self, documents): raise NotImplementedError
    def embed_query(self, query): raise NotImplementedError


class HashEmbedder(Embedder):
    def __init__(self, dimension=256): self.name = f"hash-{dimension}"; self.dimension = dimension; self.normalization_policy = "l2"
    def _one(self, text):
        v = np.zeros(self.dimension, dtype=np.float32)
        for tok in text.casefold().split():
            h = int(hashlib.sha256(tok.encode()).hexdigest()[:16], 16); i = h % self.dimension; v[i] += 1.0 if h % 2 else -1.0
        n = np.linalg.norm(v); return v / n if n else v
    def embed_documents(self, documents): return np.vstack([self._one(x) for x in documents]) if documents else np.empty((0, self.dimension), np.float32)
    def embed_query(self, query): return self._one(query)


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name, *, local_files_only: bool = False):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("sentence-transformers is required for real dense retrieval; install '.[ml]'") from exc
        self.model = SentenceTransformer(model_name, local_files_only=local_files_only)
        self.name = model_name; self.dimension = int(self.model.get_sentence_embedding_dimension()); self.normalization_policy = "l2"
    def embed_documents(self, documents):
        return self.model.encode(documents, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False, batch_size=32)
    def embed_query(self, query):
        if hasattr(self.model, "encode_query"):
            return self.model.encode_query(query, normalize_embeddings=True, convert_to_numpy=True)
        return self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)[0]


def create_embedder(name: str) -> Embedder:
    if name.startswith("hash-"):
        return HashEmbedder(int(name.split("-", 1)[1]))
    return SentenceTransformerEmbedder(name)


def embed_documents_cached(embedder, documents, cache=None):
    if cache is None:
        return embedder.embed_documents(documents)
    result = np.empty((len(documents), embedder.dimension), dtype=np.float32); missing = []
    for i, text in enumerate(documents):
        from ..cache import embedding_key
        cached = cache.get(embedding_key(text, embedder.name))
        if cached is None: missing.append((i, text))
        else: result[i] = np.asarray(cached, dtype=np.float32)
    if missing:
        vals = embedder.embed_documents([text for _, text in missing])
        for (i, text), value in zip(missing, vals):
            result[i] = value; cache.set(embedding_key(text, embedder.name), np.asarray(value).tolist())
    return result
