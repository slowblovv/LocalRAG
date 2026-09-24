# Evaluation methodology

`evaluation/queries.json` contains 60 curated retrieval queries resolved to deterministic chunk IDs in `evaluation/relevance.json`. The set covers lexical, semantic/paraphrased, distractor-heavy, and no-answer behavior. Retrieval metrics are Recall@K, Precision@K, MRR, NDCG@K, and Hit Rate@K.

The **main evaluation profile uses the configured real SentenceTransformer embedding model and Cross-Encoder reranker**. The deterministic `HashEmbedder` and `OverlapReranker` exist only as offline/unit-test doubles so CI never depends on a GPU or model download.

`scripts/evaluate.py` measures BM25, Dense, Hybrid RRF, and Hybrid + Cross-Encoder on the same fixture. `scripts/ablation.py` additionally compares Hybrid + Overlap baseline versus Hybrid + Cross-Encoder, exposing the quality/latency trade-off rather than assuming reranking is beneficial.

`evaluation/answers.json` contains a smaller controlled answer set. `scripts/evaluate_answers.py` evaluates claims against cited sentences and evidence against ground-truth relevant chunks containing expected evidence terms. It reports claim correctness, citation coverage/validity, groundedness, and refusal precision/recall. These are reproducible fixture metrics, not objective human-judge scores.

`evaluation/no_answer.json` contains 10 unsupported questions. Correct refusal is measured separately from answerable questions to expose both false-answer and over-refusal behavior.

`scripts/context_ablation.py` compares 2/4/8/12 context chunks on claim correctness, citation validity, groundedness, latency, and context size.

`scripts/benchmark.py` measures BM25, dense, hybrid and hybrid+Cross-Encoder p50/p95 retrieval latency at 1k/10k/100k synthetic chunk scales. The synthetic scaling reuses embeddings produced by the configured real model; embedding throughput is measured separately on real encoded fixture text. Index memory and serialized disk footprint are measured from the actual benchmark structures.

All benchmark values are generated from actual runs. No performance or quality number is committed as a claim in the repository.

The deterministic profile is not the main portfolio benchmark; it exists to prove pipeline behavior when external model artifacts are unavailable.
