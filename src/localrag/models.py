from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import hashlib

@dataclass(frozen=True)
class Chunk:
    id: str
    source_id: str
    text: str
    page_number: int | None
    section: str | None
    start_offset: int | None = None
    end_offset: int | None = None
    @staticmethod
    def make(document_sha256:str, index:int, text:str, source_id:str, page_number:int|None=None, section:str|None=None, start_offset:int|None=None, end_offset:int|None=None):
        content_hash=hashlib.sha256(text.encode()).hexdigest()
        cid=hashlib.sha256(f"{document_sha256}{index}{content_hash}".encode()).hexdigest()
        return Chunk(cid,source_id,text,page_number,section,start_offset,end_offset)
    def to_dict(self): return asdict(self)

@dataclass
class Document:
    id: str
    path: str
    sha256: str
    filename: str
    type: str
    size: int
    modified_at: float
    title: str | None
    created_at: str

@dataclass(frozen=True)
class SearchResult:
    chunk_id: str
    score: float
    source: str
    text: str
    metadata: dict

@dataclass
class QueryTrace:
    request_id: str
    query: str
    index_version: str
    retrieval_mode: str
    retrieved_chunk_ids: list[str]
    reranked_chunk_ids: list[str]
    context_chunk_ids: list[str]
    embedding_model: str
    reranker_model: str
    llm_model: str
    prompt_version: str
    latencies_ms: dict[str,float]
    citation_validation_result: str
    created_at: str = ''
    def __post_init__(self):
        if not self.created_at: self.created_at=datetime.now(timezone.utc).isoformat()
