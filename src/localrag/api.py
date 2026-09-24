from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from .config import settings
from .engine import RagEngine
from .index import LocalIndex
from .jobs import TinyQueueAdapter
from .metrics import index_jobs_total

app = FastAPI(title="LocalRAG", version="0.2.0")


class IndexRequest(BaseModel):
    path: str = "./knowledge"


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4096)
    top_k: int = Field(default=8, ge=1, le=50)
    retrieval_mode: str = "hybrid"


class QueryRequest(SearchRequest):
    generate: bool = True


def safe_root(path: str) -> Path:
    base = Path(settings.data_root).resolve()
    raw = Path(path)
    if raw.is_absolute():
        candidate = raw.resolve()
    elif raw in {Path("."), Path("./knowledge"), Path(settings.data_root).name}:
        candidate = base
    else:
        candidate = (base / raw).resolve()
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise HTTPException(400, "path outside DATA_ROOT") from exc
    return candidate


@app.get("/health")
def health(): return {"status": "ok"}


@app.get("/ready")
def ready():
    current = Path(settings.index_root) / "CURRENT"
    return {"status": "ready", "index_version": current.read_text().strip() if current.exists() else None}


@app.post("/api/v1/index")
def build_index(req: IndexRequest):
    root = safe_root(req.path)
    job_id = TinyQueueAdapter().submit_index(root)
    index_jobs_total.labels(status="accepted").inc()
    return {"status": "accepted", "job_id": job_id}


@app.get("/api/v1/index/status")
def index_status():
    current = Path(settings.index_root) / "CURRENT"
    status = {"index_version": current.read_text().strip() if current.exists() else None}
    try:
        jobs = TinyQueueAdapter()
        # latest local job is surfaced when the local queue is in use.
        with jobs.store._connect() as con:
            row = con.execute("SELECT id,status,last_error FROM jobs ORDER BY created_at DESC LIMIT 1").fetchone()
        if row: status["job"] = {"id": row[0], "status": row[1], "last_error": row[2]}
    except Exception:
        pass
    return status


@app.get("/api/v1/jobs/{job_id}")
def job_status(job_id: str):
    job = TinyQueueAdapter().get(job_id)
    if not job: raise HTTPException(404, "job not found")
    return job.__dict__


@app.post("/api/v1/search")
def search(req: SearchRequest):
    try:
        engine = RagEngine(); return {"results": [r.__dict__ for r in engine.search(req.query, req.retrieval_mode, min(req.top_k, settings.max_top_k))]}
    except Exception as exc: raise HTTPException(400, str(exc)) from exc


@app.post("/api/v1/query")
def query(req: QueryRequest):
    try:
        return RagEngine().query(req.query, req.retrieval_mode, min(req.top_k, settings.max_top_k), req.generate)
    except Exception as exc: raise HTTPException(400, str(exc)) from exc


@app.get("/api/v1/documents/{document_id}")
def document(document_id: str):
    try: idx = LocalIndex().load_current()
    except FileNotFoundError as exc: raise HTTPException(404, "no current index") from exc
    doc = idx.documents.get(document_id)
    if not doc: raise HTTPException(404, "document not found")
    return doc


@app.get("/api/v1/stats")
def stats():
    try: idx = LocalIndex().load_current()
    except FileNotFoundError: return {"index_version": None, "documents": 0, "chunks": 0}
    return {"index_version": idx.index_version, "documents": len(idx.documents), "chunks": len(idx.chunks), "embedding_model": idx.embedder.name}


@app.get("/metrics", response_class=PlainTextResponse)
def metrics(): return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)
