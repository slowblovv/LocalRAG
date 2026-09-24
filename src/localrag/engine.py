from __future__ import annotations

import re
import time
import uuid

from .cache import DiskCache, query_key
from .citations import validate_citations
from .confidence import retrieval_confidence
from .config import settings
from .index import LocalIndex
from .metrics import cache_hits, cache_misses, citation_invalid_total, generation_latency, query_requests_total, query_total_latency, refusal_total, reranking_latency, retrieval_latency, search_requests_total
from .models import QueryTrace
from .providers.llm import LLMProvider, create_provider
from .providers.prompt import build_user_prompt, grounded_system_prompt
from .retrieval.context import build_context
from .retrieval.hybrid import reciprocal_rank_fusion
from .retrieval.rerank import Reranker, create_reranker, rerank
from .retrieval.tokenizer import normalize_query


class RagEngine:
    engine_version = "v3"

    def __init__(self, index: LocalIndex | None = None, *, reranker: Reranker | None = None, llm: LLMProvider | None = None):
        self.index = index or LocalIndex().load_current()
        self.reranker = reranker or create_reranker(settings.reranker_model)
        self.llm = llm or create_provider(settings.llm_provider, settings.llm_model)
        self.query_cache = DiskCache(self.index.root / "cache" / "queries", settings.cache_ttl)

    def search(self, query, mode="hybrid", top_k=8):
        if len(query.encode("utf8")) > settings.max_query_length:
            raise ValueError("query too long")
        if mode not in {"lexical", "vector", "hybrid"}:
            raise ValueError("invalid retrieval_mode")
        top_k = min(max(1, top_k), settings.max_top_k)
        lexical = self.index.lexical.search(query, settings.top_k_lexical) if mode in {"lexical", "hybrid"} else []
        vector = self.index.vector.search(query, settings.top_k_vector, self.index.chunks) if mode in {"vector", "hybrid"} else []
        search_requests_total.labels(mode=mode).inc()
        if mode == "lexical": return lexical[:top_k]
        if mode == "vector": return vector[:top_k]
        return reciprocal_rank_fusion([lexical, vector], settings.rrf_k, top_k=top_k)

    def query(self, query, retrieval_mode="hybrid", top_k=8, generate=True):
        request_id = str(uuid.uuid4())
        query_requests_total.inc()
        t0 = time.perf_counter()
        cache_identity = {
            "engine_version": self.engine_version,
            "retrieval_mode": retrieval_mode,
            "top_k": top_k,
            "embedding_model": self.index.embedder.name,
            "reranker_model": self.reranker.name,
            "llm_provider": settings.llm_provider,
            "llm_model": settings.llm_model,
            "prompt_version": settings.prompt_version,
            "max_context_tokens": settings.max_context_tokens,
            "max_context_chunks": settings.max_context_chunks,
            "refuse_top_score": settings.refuse_top_score,
        }
        cache_key = query_key(normalize_query(query), cache_identity, self.index.index_version)
        cached = self.query_cache.get(cache_key) if generate else None
        if cached is not None:
            cache_hits.inc()
            trace = dict(cached.get("trace") or {})
            trace["request_id"] = request_id
            trace["cache_hit"] = True
            return {**cached, "trace": trace, "request_id": request_id, "cache_hit": True}
        cache_misses.inc()

        retrieval_start = time.perf_counter()
        candidates = self.search(query, retrieval_mode, max(top_k, settings.top_k_rerank))
        retrieval_seconds = time.perf_counter() - retrieval_start
        retrieval_latency.observe(retrieval_seconds)
        lexical = self.index.lexical.search(query, settings.top_k_lexical) if retrieval_mode in {"lexical", "hybrid"} else []
        vector = self.index.vector.search(query, settings.top_k_vector, self.index.chunks) if retrieval_mode in {"vector", "hybrid"} else []

        rerank_start = time.perf_counter()
        reranked = rerank(query, candidates, self.reranker, min(settings.top_k_rerank, top_k))
        rerank_seconds = time.perf_counter() - rerank_start
        reranking_latency.observe(rerank_seconds)
        context_start = time.perf_counter()
        context, context_token_count = build_context(reranked, settings.max_context_tokens, min(settings.max_context_chunks, top_k))
        context_seconds = time.perf_counter() - context_start
        confidence = retrieval_confidence(query, reranked, lexical, vector)

        generation_seconds = 0.0; refused = False; citation_ids: list[int] = []; answer = ""; llm_model = settings.llm_model
        if confidence["score"] < settings.refuse_top_score or not context:
            answer = "I don't have enough evidence in the indexed documents to answer this question."
            refused = True
            refusal_total.inc()
        elif not generate:
            answer = ""
        else:
            system = grounded_system_prompt(settings.prompt_version)
            user = build_user_prompt(query, context)
            generation_start = time.perf_counter()
            response = self.llm.generate(system, user, 512, 0.0)
            generation_seconds = time.perf_counter() - generation_start
            generation_latency.observe(generation_seconds)
            answer = response.text; llm_model = response.model
            citation_ids = [int(x) for x in re.findall(r"\[(\d+)\]", answer)]

        valid_ids, citation_ok = validate_citations(answer, len(context))
        invalid_ids = [i for i in citation_ids if i not in valid_ids]
        if invalid_ids:
            citation_invalid_total.inc(len(invalid_ids))
            answer = ""
            citation_result = "invalid"
        elif refused:
            citation_result = "refused"
        elif citation_ids and citation_ok:
            citation_result = "valid"
        else:
            citation_result = "none"

        citations = []
        for i in valid_ids:
            c = context[i - 1]
            citations.append({
                "id": str(i),
                "document": c.metadata.get("filename") or c.metadata.get("document") or c.metadata.get("source_id"),
                "source_id": c.metadata.get("source_id"),
                "page": c.metadata.get("page_number"),
                "section": c.metadata.get("section"),
                "chunk_id": c.chunk_id,
            })

        total_seconds = time.perf_counter() - t0
        query_total_latency.observe(total_seconds)
        trace = QueryTrace(
            request_id=request_id,
            query=query,
            index_version=self.index.index_version,
            retrieval_mode=retrieval_mode,
            retrieved_chunk_ids=[r.chunk_id for r in candidates],
            reranked_chunk_ids=[r.chunk_id for r in reranked],
            context_chunk_ids=[r.chunk_id for r in context],
            embedding_model=self.index.embedder.name,
            reranker_model=self.reranker.name,
            llm_model=llm_model,
            prompt_version=settings.prompt_version,
            latencies_ms={"retrieval": retrieval_seconds * 1000, "reranking": rerank_seconds * 1000, "context": context_seconds * 1000, "generation": generation_seconds * 1000},
            citation_validation_result=citation_result,
        )
        response = {
            "request_id": request_id,
            "answer": answer,
            "citations": citations,
            "retrieval": {
                "lexical_candidates": len(lexical),
                "vector_candidates": len(vector),
                "reranked_candidates": len(reranked),
                "context_chunks": len(context),
                "context_tokens": context_token_count,
            },
            "latency_ms": {**trace.latencies_ms, "total": total_seconds * 1000},
            "index_version": self.index.index_version,
            "embedding_model": self.index.embedder.name,
            "reranker_model": self.reranker.name,
            "llm_model": llm_model,
            "prompt_version": settings.prompt_version,
            "trace": {**trace.__dict__, "retrieval_confidence": confidence, "cache_hit": False},
            "cache_hit": False,
        }
        if generate and citation_result in {"valid", "refused"}:
            self.query_cache.set(cache_key, response)
        return response
