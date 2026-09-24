from __future__ import annotations

import argparse
import os

from localrag.index import LocalIndex
from localrag.engine import RagEngine
from localrag.retrieval.embed import HashEmbedder
from localrag.retrieval.rerank import OverlapReranker
from localrag.providers.llm import FakeLLMProvider


def main():
    parser = argparse.ArgumentParser(description="LocalRAG reproducible demo")
    parser.add_argument("--deterministic", action="store_true", help="use CI/test doubles; main portfolio profile uses real models")
    args = parser.parse_args()
    if args.deterministic:
        os.environ["EMBEDDING_MODEL"] = "hash-256"
        os.environ["RERANKER_MODEL"] = "overlap"
        os.environ["LLM_PROVIDER"] = "fake"
    idx = LocalIndex(embedder=HashEmbedder(256) if args.deterministic else None)
    info = idx.build("demo_corpus")
    idx.load_current()
    engine = RagEngine(
        idx,
        reranker=OverlapReranker() if args.deterministic else None,
        llm=FakeLLMProvider() if args.deterministic else None,
    )
    print("INDEX", info)
    for q in ["What does TCP provide?", "Why can p99 matter more than mean latency?", "What is the salary of the networking engineer?"]:
        r = engine.query(q, "hybrid", 8, True)
        print("\nQ:", q)
        print("A:", r["answer"])
        print("CITATIONS:", r["citations"])
        print("TRACE:", r["latency_ms"])


if __name__ == "__main__":
    main()
