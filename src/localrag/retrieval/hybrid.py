from __future__ import annotations
from ..models import SearchResult

def reciprocal_rank_fusion(result_lists:list[list[SearchResult]],k:int=60,top_k:int=20)->list[SearchResult]:
    scores={}; best={}
    for results in result_lists:
        for rank,r in enumerate(results,1):
            scores[r.chunk_id]=scores.get(r.chunk_id,0)+1/(k+rank); best.setdefault(r.chunk_id,r)
    out=[]
    for cid,score in sorted(scores.items(),key=lambda kv:kv[1],reverse=True)[:top_k]:
        r=best[cid]; out.append(SearchResult(r.chunk_id,float(score),'hybrid',r.text,r.metadata))
    return out
