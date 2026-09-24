from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
from ..models import Chunk, SearchResult

class VectorIndex:
    def __init__(self,embedder): self.embedder=embedder; self.vectors=np.empty((0,embedder.dimension),dtype=np.float32); self.chunk_ids=[]; self.index_version=''
    def build(self,chunks,index_version):
        self.chunk_ids=[c.id for c in chunks]; self.vectors=self.embedder.embed_documents([c.text for c in chunks]); self.index_version=index_version
    def search(self,query,top_k,chunks_by_id):
        if not len(self.vectors): return []
        q=self.embedder.embed_query(query); scores=self.vectors@q; order=np.argsort(-scores)[:top_k]
        return [SearchResult(self.chunk_ids[i],float(scores[i]),'vector',chunks_by_id[self.chunk_ids[i]].text,chunks_by_id[self.chunk_ids[i]].to_dict()) for i in order]
    def save(self,root:Path):
        root.mkdir(parents=True,exist_ok=True); np.save(root/'vectors.npy',self.vectors); (root/'meta.json').write_text(json.dumps({'chunk_ids':self.chunk_ids,'index_version':self.index_version,'embedding_model':self.embedder.name,'embedding_dimension':self.embedder.dimension,'normalization_policy':self.embedder.normalization_policy}),encoding='utf8')
    @classmethod
    def load(cls,root:Path,embedder):
        meta=json.loads((root/'meta.json').read_text());
        if meta['embedding_model']!=embedder.name or int(meta['embedding_dimension'])!=embedder.dimension or meta['normalization_policy']!=embedder.normalization_policy: raise ValueError('ERR incompatible vector index')
        x=cls(embedder); x.vectors=np.load(root/'vectors.npy',mmap_mode='r'); x.chunk_ids=meta['chunk_ids']; x.index_version=meta['index_version']; return x
