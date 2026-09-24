from __future__ import annotations

import argparse
import json
from pathlib import Path

from localrag.config import settings
from localrag.engine import RagEngine
from localrag.index import LocalIndex
from localrag.retrieval.rerank import create_reranker
from scripts.evaluate_answers import claim_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deterministic", action="store_true", help="test-double mode for CI")
    _args = parser.parse_args()
    idx = LocalIndex(); idx.build('demo_corpus'); idx.load_current()
    engine = RagEngine(idx, reranker=create_reranker(settings.reranker_model))
    items = json.loads(Path('evaluation/answers.json').read_text())
    rel = json.loads(Path('evaluation/relevance.json').read_text())[::4]
    original = settings.max_context_chunks
    try:
        print('Context chunks | Correctness | Citation validity | Groundedness | mean latency ms | mean context tokens')
        for n in [2, 4, 8, 12]:
            settings.max_context_chunks = n
            correctness = []; validity = []; grounded = []; latencies = []; tokens = []
            for item, relevance in zip(items, rel):
                result = engine.query(item['query'], 'hybrid', min(n, 10), True)
                ctx = {cid: idx.chunks[cid].text for cid in result['trace']['context_chunk_ids']}
                metrics = claim_metrics(result['answer'], result['citations'], item['claims'], set(relevance['relevant_chunk_ids']), ctx)
                correctness.append(metrics['claim_correctness'])
                validity.append(metrics['citation_validity'])
                grounded.append(metrics['evidence_support'])
                latencies.append(result['latency_ms']['total'])
                tokens.append(result['retrieval']['context_tokens'])
            print(f"{n} | {sum(correctness)/len(correctness):.4f} | {sum(validity)/len(validity):.4f} | {sum(grounded)/len(grounded):.4f} | {sum(latencies)/len(latencies):.3f} | {sum(tokens)/len(tokens):.1f}")
    finally:
        settings.max_context_chunks = original


if __name__ == '__main__': main()
