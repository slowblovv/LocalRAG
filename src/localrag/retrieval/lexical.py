from __future__ import annotations
import importlib
from .bm25 import BM25Index
from ..models import Chunk, SearchResult

class TinySearchAdapter:
    """Adapter for prior TinySearch BM25. Falls back to the reference-compatible local BM25 implementation."""
    def __init__(self, external_import:str|None=None): self.external_import=external_import; self.index=BM25Index(); self.chunks={}
    def build(self,chunks): self.chunks={c.id:c for c in chunks}; self.index.build(chunks)
    def search(self,query,top_k=20):
        return [SearchResult(cid,score,'lexical',self.chunks[cid].text,self.chunks[cid].to_dict()) for cid,score in self.index.search(query,top_k)]
    def dump(self,path): self.index.dump(path)
    def load(self,path,chunks): self.index=BM25Index.load(path); self.chunks={c.id:c for c in chunks}
