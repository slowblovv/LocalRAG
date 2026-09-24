from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from .cache import DiskCache
from .config import settings
from .ingest.chunker import ingest_file
from .metrics import index_chunks_total, index_documents_total
from .models import Chunk
from .retrieval.embed import Embedder, create_embedder, embed_documents_cached
from .retrieval.lexical import TinySearchAdapter
from .retrieval.vector import VectorIndex


class LocalIndex:
    def __init__(self, root: str | Path | None = None, *, embedder: Embedder | None = None):
        self.root = Path(root or settings.index_root)
        self.chunks: dict[str, Chunk] = {}
        self.documents: dict[str, dict] = {}
        self.lexical = TinySearchAdapter()
        self.embedder = embedder or create_embedder(settings.embedding_model)
        self.vector = VectorIndex(self.embedder)
        self.index_version = ""
        self.embedding_cache = DiskCache(self.root / "cache" / "embeddings", settings.cache_ttl)

    def build(self, data_root: str | Path | None = None) -> dict:
        root = Path(data_root or settings.data_root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        files = []
        for p in sorted(root.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in {".md", ".txt", ".pdf"}:
                continue
            if p.stat().st_size > settings.max_document_size:
                raise ValueError(f"document exceeds max size: {p.name}")
            files.append(p)

        previous = None; previous_chunks: dict[str, Chunk] = {}; previous_docs: dict[str, dict] = {}
        current = self.root / "CURRENT"
        if current.exists():
            pv = current.read_text().strip(); old_root = self.root / pv
            try:
                previous_docs = {d["path"]: d for d in json.loads((old_root / "documents.json").read_text())}
                previous_chunks = {r["id"]: Chunk(**r) for r in json.loads((old_root / "chunks.json").read_text())}
                previous = old_root
            except (OSError, ValueError, KeyError):
                previous = None

        chunks: list[Chunk] = []; docs: list[dict] = []; seen_sha: set[str] = set()
        for p in files:
            raw = p.read_bytes(); sha = hashlib.sha256(raw).hexdigest()
            if sha in seen_sha:
                continue  # explicit repository policy: content-identical files are deduplicated.
            seen_sha.add(sha)
            doc = {"id": sha, "path": str(p), "sha256": sha, "filename": p.name, "type": p.suffix.lower(), "size": p.stat().st_size, "modified_at": p.stat().st_mtime, "title": p.stem}
            docs.append(doc)
            old = previous_docs.get(str(p.resolve())) if previous_docs else None
            if old and old.get("sha256") == sha and previous:
                same = [c for c in previous_chunks.values() if c.source_id == old["id"]]
                if same:
                    chunks.extend(same); continue
            _, _, new_chunks = ingest_file(p)
            chunks.extend(new_chunks)

        version = hashlib.sha256("".join(c.id for c in chunks).encode()).hexdigest()[:12]
        tmp = self.root / f".tmp-{version}"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True, exist_ok=True)
        (tmp / "chunks.json").write_text(json.dumps([c.to_dict() for c in chunks], ensure_ascii=False), encoding="utf8")
        self.chunks = {c.id: c for c in chunks}; self.lexical.build(chunks); self.lexical.dump(tmp / "bm25.json")
        self.vector.vectors = embed_documents_cached(self.embedder, [c.text for c in chunks], self.embedding_cache)
        self.vector.chunk_ids = [c.id for c in chunks]; self.vector.index_version = version; self.vector.save(tmp / "vector")
        self.documents = {d["id"]: d for d in docs}; (tmp / "documents.json").write_text(json.dumps(docs, ensure_ascii=False), encoding="utf8")
        target = self.root / version
        if target.exists(): shutil.rmtree(target)
        tmp.rename(target); (self.root / "CURRENT").write_text(version); self.index_version = version
        index_documents_total.inc(len(docs)); index_chunks_total.inc(len(chunks))
        return {"index_version": version, "documents": len(docs), "chunks": len(chunks), "embedding_model": self.embedder.name}

    def load_current(self):
        current = self.root / "CURRENT"
        if not current.exists():
            raise FileNotFoundError("no current index")
        version = current.read_text().strip(); target = self.root / version
        rows = json.loads((target / "chunks.json").read_text()); self.chunks = {r["id"]: Chunk(**r) for r in rows}
        self.documents = {d["id"]: d for d in json.loads((target / "documents.json").read_text())}
        self.lexical.load(target / "bm25.json", list(self.chunks.values())); self.vector = VectorIndex.load(target / "vector", self.embedder)
        if self.vector.index_version != version:
            raise ValueError("ERR vector index version mismatch")
        self.index_version = version
        return self
