# Security model

The document root is an explicit sandbox. Relative index paths are resolved under `DATA_ROOT`, and path traversal or absolute paths outside the configured root are rejected. Query length, top-k, context size, document size, and indexing scope are bounded.

Retrieved content is **untrusted data**, never instructions. The prompt contract separates system rules, the user question, and document context. Document text is escaped before insertion, embedded instructions are explicitly ignored, and the v1 pipeline exposes no arbitrary tool execution.

The repository contains an end-to-end malicious-document test that exercises retrieval → context construction → LLM boundary → citation validation. A fake adversarial model attempts a tool-looking action if it is outside the untrusted-data boundary; the test verifies that the retrieved document cannot trigger it.

Vector indexes are loaded only after compatibility checks for embedding model, dimension, normalization policy, and index version. Model loading is restricted to configured local/model-runtime abstractions; documents are never implicitly uploaded to a cloud API.

Content-identical documents are deduplicated by SHA-256 during ingestion. This is an explicit repository policy: two files with identical bytes produce one indexed document rather than two indistinguishable evidence sources.
