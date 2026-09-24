# LocalRAG

Local-first, citation-grounded document QA with lexical BM25 retrieval, real dense embeddings, RRF fusion, Cross-Encoder reranking, context budgeting, a local LLM provider boundary, server-side citation validation, refusal logic, asynchronous indexing, and evaluation/benchmark tooling.

This project is intentionally not a generic “PDF → embeddings → LLM” demo. Its primary artifact is an evaluation-driven retrieval pipeline with an explicit quality/latency trade-off.

## Pipeline

```text
Documents
   ↓
parser → deterministic chunks
   ├────────→ TinySearch/BM25 ────┐
   └────────→ SentenceTransformer ─┤
                                   ↓
                              RRF fusion
                                   ↓
                         Cross-Encoder reranker
                                   ↓
                         context builder/budget
                                   ↓
                              local LLM
                                   ↓
                       citation validation/refusal
```

## Quickstart

```bash
docker compose up --build
curl http://localhost:8000/health
curl -X POST http://localhost:8000/api/v1/index -H 'content-type: application/json' -d '{"path":"./knowledge"}'
```

Indexing is asynchronous in the server path:

```text
POST /api/v1/index
      ↓
job_id
      ↓
TinyQueueAdapter / durable local queue
      ↓
localrag.worker
      ↓
atomic index build + validation + CURRENT swap
```

For an actual local model, configure the Ollama provider. The default architecture never sends documents to a cloud model implicitly.

For an offline CI-style smoke test with deterministic doubles:

```bash
python -m scripts.demo --deterministic
```

## Documents and duplicate policy

Mandatory formats: `.md`, `.txt`, `.pdf`. PDFs preserve page numbers when available.

Content-identical documents are deduplicated by SHA-256. This is an explicit repository policy: identical bytes map to one indexed source so retrieval and citation provenance do not double-count duplicate files. Changed documents are re-chunked/re-embedded; unchanged content reuses deterministic chunks/cache entries; deleted documents disappear from the new index.

## Retrieval

The API supports `lexical`, `vector`, and `hybrid` modes. Hybrid uses Reciprocal Rank Fusion:

`RRF(d) = sum(1 / (k + rank_i(d)))`

The standard main/evaluation profile uses:

- SentenceTransformer: `sentence-transformers/all-MiniLM-L6-v2`
- Cross-Encoder: `cross-encoder/ms-marco-MiniLM-L6-v2`
- 20 lexical + 20 dense candidates
- 40 fused candidates to the reranker
- 10 final reranked/context candidates

The Cross-Encoder is a separate replaceable interface. `OverlapReranker` is retained only as a measured baseline and deterministic test double.

## TinySearch reuse

`localrag.retrieval.lexical.TinySearchAdapter` is the lexical integration boundary for the earlier TinySearch project. A documented external implementation hook can be connected through `TINYSEARCH_IMPORT`; the repository includes a behaviorally compatible BM25 fallback for standalone operation and CI.

## Embeddings and vector compatibility

`SentenceTransformerEmbedder` is the production dense-retrieval path. Its embedding normalization is explicit and persisted with the index metadata. The index loader rejects incompatible embedding model, dimension, normalization policy, or index-version metadata.

`HashEmbedder` is intentionally scoped to deterministic unit/integration tests where model downloads or GPU availability would make CI nondeterministic. Main benchmarks and quality evaluations explicitly reject the hash embedder.

## Grounding and safety

Retrieved text is untrusted document data, never instructions. The prompt separates system rules, the user question, and document context; document text is escaped before insertion. The application does not expose arbitrary tools to retrieved documents.

Citation IDs are validated server-side and must correspond to retrieved chunks. Invalid citations invalidate the answer instead of being silently trusted. Low-confidence/no-evidence queries are refused rather than answered from unsupported model knowledge.

The test suite contains an end-to-end malicious-document scenario covering retrieval → context → LLM boundary → citation validation.

## Evaluation

The repository contains a deterministic fixture corpus plus 60 retrieval queries and a smaller claim/evidence answer set.

Run the main quality evaluation:

```bash
python -m scripts.prepare_eval
python -m scripts.evaluate
python -m scripts.evaluate_answers
```

Compare retrieval/reranking stages:

```bash
python -m scripts.ablation
python -m scripts.context_ablation
```

The ablation table compares BM25, Dense, Hybrid RRF, Hybrid + Overlap baseline, and Hybrid + Cross-Encoder. It reports quality together with latency so the reranker is evaluated as a quality/latency trade-off rather than assumed to be universally better.

Answer evaluation is claim/evidence based: a generated claim must appear in a cited sentence to count as correct, while groundedness additionally requires the citation to map to a ground-truth relevant chunk containing expected evidence terms. Automated scores are reproducible fixture proxies, not objective human or LLM-judge truth.

No-answer evaluation contains 10 unsupported queries and reports refusal precision/recall.

## Performance benchmark

`python -m scripts.benchmark` is a real-model benchmark. It loads the configured SentenceTransformer and Cross-Encoder and reports measured p50/p95 latency at 1k/10k/100k synthetic chunk scales, plus real embedding throughput, RAM footprint, and serialized index size.

The scaling harness reuses vectors produced by the configured real model for the synthetic scale expansion; embedding throughput is separately measured on real fixture text. No performance number is committed as a project claim.

## Observability

Prometheus `/metrics` reports actual counters/histograms for indexing, search/query traffic, retrieval/reranking/generation latency, citation failures, refusals, cache hits/misses, and indexing job lifecycle. Query responses include request ID, index/embedding/reranker/LLM/prompt versions and a compact trace.

Cache hits receive a fresh `request_id`; cached response identity includes retrieval configuration and model/index/prompt identity, so a new request is not confused with the original trace.

## API

Required endpoints are implemented under `/api/v1`:

```text
POST /index
GET  /index/status
GET  /jobs/{job_id}
POST /search
POST /query
GET  /documents/{id}
GET  /stats
GET  /metrics
GET  /health
GET  /ready
```

## Configuration

Important environment variables include:

```text
DATA_ROOT
INDEX_ROOT
JOB_DB_PATH
EMBEDDING_MODEL
VECTOR_INDEX_TYPE
RRF_K
RERANKER_MODEL
LLM_PROVIDER
LLM_MODEL
MAX_CONTEXT_TOKENS
TOP_K_LEXICAL
TOP_K_VECTOR
TOP_K_RERANK
CACHE_TTL
TINYQUEUE_IMPORT
```

## CI

CI uses deterministic hash embeddings and fake LLMs for reproducible tests and runs the full security/citation/retrieval fixture suite without GPU access. Real-model benchmark/evaluation commands remain available for developer environments with the ML dependencies and model artifacts installed.

## Limitations

The separate TinyQueue project is not bundled, so `TinyQueueAdapter` documents the external enqueue contract and provides a durable local queue fallback. The bundled local vector index is single-node. The Cross-Encoder adds latency intentionally; benchmark output should be used to select an appropriate candidate/context budget. No agent tool execution is part of v1.
