from __future__ import annotations
import re
from ..models import SearchResult


def _estimate_tokens(text): return len(re.findall(r"\S+", text))
def _jaccard(a,b):
    sa=set(re.findall(r"\w+",a.casefold())); sb=set(re.findall(r"\w+",b.casefold())); return len(sa&sb)/max(len(sa|sb),1)

def build_context(results:list[SearchResult], max_tokens=6000, max_chunks=20, max_per_source=3, redundancy_threshold=.85):
    chosen=[]; used=set(); total=0; source_counts={}
    for r in results:
        if r.chunk_id in used: continue
        if any(_jaccard(r.text,c.text)>=redundancy_threshold for c in chosen): continue
        source = r.metadata.get('source_id') or r.metadata.get('document') or 'unknown'
        if source_counts.get(str(source),0)>=max_per_source: continue
        n=_estimate_tokens(r.text)
        if total+n>max_tokens: continue
        chosen.append(r); used.add(r.chunk_id); total+=n; source_counts[str(source)]=source_counts.get(str(source),0)+1
        if len(chosen)>=max_chunks: break
    return chosen,total
