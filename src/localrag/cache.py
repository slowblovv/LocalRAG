from __future__ import annotations
import hashlib, json, time
from pathlib import Path

class DiskCache:
    def __init__(self, root, ttl=300): self.root=Path(root); self.ttl=ttl; self.root.mkdir(parents=True,exist_ok=True)
    def _path(self,key): return self.root/(hashlib.sha256(key.encode()).hexdigest()+'.json')
    def get(self,key):
        p=self._path(key)
        if not p.exists(): return None
        try:
            d=json.loads(p.read_text())
            if time.time()-d['created_at']>self.ttl: p.unlink(missing_ok=True); return None
            return d['value']
        except (OSError,ValueError,KeyError): return None
    def set(self,key,value):
        p=self._path(key); tmp=p.with_suffix('.tmp'); tmp.write_text(json.dumps({'created_at':time.time(),'value':value}),encoding='utf8'); tmp.replace(p)

def embedding_key(chunk_text, model_version): return hashlib.sha256((chunk_text+model_version).encode()).hexdigest()
def query_key(normalized_query, retrieval_config, index_version): return hashlib.sha256((normalized_query+json.dumps(retrieval_config,sort_keys=True)+index_version).encode()).hexdigest()
